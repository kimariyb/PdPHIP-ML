import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import pandas as pd
from tqdm import tqdm

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.inchi import MolToInchiKey  # 去重推荐用 InChIKey


# 关闭 RDKit 的报警输出，处理海量脏数据时很有必要
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")

# 允许的元素集合 (有机小分子，避免金属有机物干扰)
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "F", "Cl", "Br"}

ALKYNE_PATTERN = Chem.MolFromSmarts("C#C")
if ALKYNE_PATTERN is None:
    raise ValueError("ALKYNE_PATTERN SMARTS 编译失败")


@dataclass(frozen=True)
class FilterConfig:
    max_mw: float = 300.0
    min_c: int = 4
    max_c: int = 12
    min_heavy: int = 4
    max_heavy: int = 15
    min_logp: float = -1.0
    max_logp: float = 4.0
    max_rgroup_c: int = 2          # 三键任一端 R 基团最大碳数（甲基/乙基=2）
    require_single_alkyne: bool = True
    exclude_symmetric_alkyne: bool = True


def is_symmetric_alkyne(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """
    判断指定三键是否对称：比较键两端原子在分子中的 canonical rank。
    """
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    return ranks[a1] == ranks[a2]


def check_alkyne_rgroup_size(mol: Chem.Mol, bond_idx: int, max_r_c: int) -> bool:
    """
    判断指定三键两侧是否存在一侧满足：R 基团碳数 <= max_r_c。
    用 FragmentOnBonds 断键成两段，计算每段碳数。
    每段都包含一个炔碳 => R碳数 = 该段碳数 - 1。
    """
    try:
        frags_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
        frags = Chem.GetMolFrags(frags_mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return False

    if len(frags) != 2:
        return False

    def carbon_count(m: Chem.Mol) -> int:
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6)

    c1 = carbon_count(frags[0])
    c2 = carbon_count(frags[1])

    # R <= max_r_c  <=>  fragment_C <= max_r_c + 1
    threshold = max_r_c + 1
    return (c1 <= threshold) or (c2 <= threshold)


def ok_smiles(smi: str, cfg: FilterConfig) -> Tuple[Optional[str], Optional[str]]:
    """
    通过则返回 (canonical_smiles, None)
    不通过返回 (None, reason)
    """
    if not smi or not isinstance(smi, str):
        return None, "empty_or_not_str"

    smi = smi.strip()

    # --- Stage 1: 快速字符串过滤 ---
    if "#" not in smi:
        return None, "no_triple_bond_char"
    if "." in smi:
        return None, "multiple_fragments_dot"
    if "@" in smi:
        return None, "has_stereo_at"

    # --- Stage 2: 解析分子 ---
    try:
        m = Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        return None, "molfromsmiles_exception"
    if m is None:
        return None, "molfromsmiles_none"

    # 去立体信息（你原来的逻辑保留）
    Chem.RemoveStereochemistry(m)

    # --- Stage 3: 基本属性过滤 ---
    try:
        if MolWt(m) > cfg.max_mw:
            return None, "mw_too_high"
    except Exception:
        return None, "mw_calc_failed"

    for atom in m.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return None, "disallowed_element"
        if atom.GetNumRadicalElectrons() != 0:
            return None, "has_radical"
        if atom.GetIsotope() != 0:
            return None, "has_isotope"

    carbon_count = sum(1 for atom in m.GetAtoms() if atom.GetAtomicNum() == 6)
    if not (cfg.min_c <= carbon_count <= cfg.max_c):
        return None, "carbon_count_out_of_range"

    heavy_atom_count = m.GetNumHeavyAtoms()
    if not (cfg.min_heavy <= heavy_atom_count <= cfg.max_heavy):
        return None, "heavy_atom_count_out_of_range"

    try:
        lp = MolLogP(m)
        if not (cfg.min_logp <= lp <= cfg.max_logp):
            return None, "logp_out_of_range"
    except Exception:
        return None, "logp_calc_failed"

    if len(Chem.GetMolFrags(m)) != 1:
        return None, "not_single_fragment"

    if Chem.GetFormalCharge(m) != 0:
        return None, "formal_charge_nonzero"

    # --- Stage 4: 结构特征过滤 ---
    if rdMolDescriptors.CalcNumBridgeheadAtoms(m) > 0:
        return None, "has_bridgehead"
    if rdMolDescriptors.CalcNumSpiroAtoms(m) > 0:
        return None, "has_spiro"

    # 环大小过滤
    # 使用RDKit的环信息检测三元环和8元环以上的环
    ring_info = m.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_size = len(ring)
        # 过滤容易开环的3元环和4元环
        if ring_size <= 4:
            return None, "has_small_ring"
        # 过滤大环
        if ring_size >= 8:
            return None, "has_large_ring"

    # --- Stage 5: 核心三键逻辑 ---
    matches = m.GetSubstructMatches(ALKYNE_PATTERN)
    if cfg.require_single_alkyne and len(matches) != 1:
        return None, "alkyne_count_not_1"

    # 若不强制唯一，这里取第一个三键也可以
    if len(matches) < 1:
        return None, "no_alkyne_match"

    idx1, idx2 = matches[0]
    bond = m.GetBondBetweenAtoms(idx1, idx2)
    if bond is None:
        return None, "bond_not_found"
    bond_idx = bond.GetIdx()

    if not check_alkyne_rgroup_size(m, bond_idx, max_r_c=cfg.max_rgroup_c):
        return None, "rgroup_too_large"

    if cfg.exclude_symmetric_alkyne and is_symmetric_alkyne(m, bond):
        return None, "symmetric_alkyne"

    # --- Stage 6: 返回标准化 SMILES ---
    try:
        can = Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None, "moltosmiles_failed"

    return can, None


def load_smiles_from_tsv(path: str) -> List[str]:
    df = pd.read_csv(path, header=None, names=["smiles", "zinc_id"], sep="\t")
    return df["smiles"].astype(str).tolist()


def dedup_by_inchikey(smiles_list: List[str]) -> List[str]:
    """
    用 InChIKey 去重：同一结构不同 SMILES 表达会合并。
    返回去重后 canonical smiles（基于 RDKit MolToSmiles）。
    """
    seen = set()
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        key = MolToInchiKey(m)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(Chem.MolToSmiles(m, canonical=True))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="ZINC_dataset.csv (tsv, smiles\\tzinc_id)")
    parser.add_argument("--output", type=str, default="filtered_unique_smiles.txt", help="输出去重后的 SMILES")
    args = parser.parse_args()

    cfg = FilterConfig()

    zinc_smiles = load_smiles_from_tsv(args.input)
    print(f"ZINC数据集共加载 {len(zinc_smiles)} 个SMILES")

    # 你的实验 SMILES（保留）
    experiment_smiles = [
        'C#Cc1ccc(F)cc1',
        'C#Cc1ccc(C=O)cc1',
        'C#Cc1ccc(CO)cc1',
        'C#Cc1ccc(OC)cc1',
        'C#Cc1ccc(C(C)(C)C)cc1',
        'C#Cc1ccccc1',
        'C#CC(C)(C)O',
        'C#CCCO',
        'C#CC(C)(C)C',
        'C#CCCCCCC',
        'C#CC1CCCCC1',
        'CC#Cc1ccccc1',
        'CC#CCC',
        'CC#CCCCC',
        'CC#CCCCCC',
        'CCCCCC#CCO',
        'CCCCCC#CCCl',
        'CCC#CC(=O)O',
    ]

    # 过滤
    reasons = Counter()
    valid_smiles = []
    for smi in tqdm(zinc_smiles, desc="筛选有效SMILES"):
        can, reason = ok_smiles(smi, cfg)
        if can is None:
            reasons[reason] += 1
        else:
            valid_smiles.append(can)

    print(f"通过过滤: {len(valid_smiles)}")
    if reasons:
        print("失败原因Top10：")
        for k, v in reasons.most_common(10):
            print(f"  {k}: {v}")

    # 合并 + 去重
    all_smiles = valid_smiles + experiment_smiles
    print(f"合并后的SMILES总数: {len(all_smiles)}")

    unique_canon_smiles = dedup_by_inchikey(all_smiles)
    print(f"最终得到的唯一规范SMILES数量(InChIKey去重): {len(unique_canon_smiles)}")

    # 输出
    with open(args.output, "w", encoding="utf-8") as f:
        for s in unique_canon_smiles:
            f.write(s + "\n")
    print(f"已写出: {args.output}")


if __name__ == "__main__":
    main()
