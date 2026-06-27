import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Crippen import MolLogP
from tqdm.auto import tqdm
from multiprocessing import Pool
from os import cpu_count

# 关闭 RDKit 的报警输出，处理海量脏数据时很有必要
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')

# 允许的元素集合 (有机小分子，避免金属有机物干扰)
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "F", "Cl", "Br"}

UNALLOWED_SMARTS = [
    "[C]=[C]",
    "[C]=[N]",
    "[C]#[N]",
    "[S,C](=[O,S])[F,Br,Cl,I]",
    "O=CN=[N+]=[N-]",
    "C(=O)OC(=O)",
    "OO",
    "OOO",
    "C(=O)Oc1c(F)c(F)c(F)c(F)c1(F)",
    "C(=O)Oc1ccc(N(=O)=O)cc1",
    "N=C=[S,O]",
    "cN=[N+]=[N-]",
    "[N;R0][N;R0]C(=O)",
    "[Cl]C([C&R0])=N",
    "[N&D2](=O)",
    "N=C=N",
    "[N+]#[C-]",
    "C(=O)N(C(=O))OC(=O)",
    "N#CC[OH]",
    "N#CC(=O)",
    "[N;R0]=[N;R0]C#N",
    "[N;R0]=[N;R0]CC=O",
    "[O;R1][C;R1][C;R1][O;R1][C;R1][C;R1][O;R1]",
    "N[CH2]C#N",
    "C1(=O)OCC1",
    "N1CCC1=O",
    "O=C1[#6]~[#6]C(=O)[#6]~[#6]1",
    "C=CC=CC=CC=C",
    "c-N(=O)~O",
    "O=C-N!@N",
    "C!@N=*",
    "[O,N,S][CH2]N1C(=O)CCC1(=O)",
    "[$(N!@N),$([N;R0]=[N;R0])]",
    "[N,n][N,n]",
    "[Cl,Br,I]-N",
    "C#C-[F,Br,I,Cl]",
    "C(-[O;H1])(-C#N)",
    "C(-O)-C-N(=O)=O",
    "C(=O)-C(=N)",
    "C(=O)-C([F,Br,I,Cl])-C(=O)",
    "C(=O)-N(-O)-C(=O)",
    "[C;!R](=O)-[N;!R]-[C;!R](=O)",
    "C(=O)-N-N(=O)",
    "C(=O)-N-N-C(=O)",
    "C(=O)-O-N-C(=O)",
    "C([F,Br,I,Cl])=N",
    "C-N=O",
    "C1-C-C(=O)-O1",
    "C=N-C(-O)",
    "C=N-C(=O)",
    "C=N-N",
    "C=N-N=C",
    "C=N-O",
    "C=N-O-C(=O)",
    "C=N=O",
    "N(=O)-C([F,Br,I,Cl])",
    "[N!H0]-[C!H0]-[N!H0]",
    "N(O)=C-C-N(=O)",
    "N-O-C(=O)",
    "N1-N-C(=O)-N-N1",
    "N=C(-N)-C(=N)-N",
    "N=C(-O)-N",
    "N=C([F,Br,I,Cl])",
    "N=C-C(=O)",
    "N=C=O",
    "O-C(=O)-O-N",
    "O-N=C-C=N-O",
    "[N;H]-[C;H]-[N;H]",
    "c-C(-O)(-O)-c",
    "c-C(-[O;H])[!O]",
    "C1-C-O-C-O-C1",
    "C(=O)-C(=N)",
    "C=N-O",
    "N(=O)(=O)",
    "N(=O)(-O)",
    "N(-O)(-O)",
    "C(=N)-C=N-O",
    "C(-C#N)(-C#N)",
    "C-O-N=O",
    "N-N=O",
    "N-C(=O)-C(=O)-N",
    "c=N",
    "C1-O-C-O-C1",
    "[C;!R](=O)-[C;!R](=O)",
    "C(=O)-N-O-C(=O)",
    "C1(=O)-C=C-C(=O)-C=C1",
    "ONO",
    "ON(~O)~O",
    "N=[N+]=N",
    "N-C#N",
    "[C;!R]=N-N=[C;!R]",
    "[C;a]-[N;H1]-[N;H2]",
    "[C;a]-[C;H2]([F,Br,I,Cl])",
    "N(C)(C)-[C;H2]-[C;H2]([F,Br,I,Cl])",
    "[N+]#N",
    "ON#C",
    "[N;!R]=[N;!R]",
    "C1NC(=O)OC(=O)C1",
    "[C;!R]C(=N)O[C;!R]",
    "C(=O)-O-C(=O)-[!N]",
    "C#C-C#N",
    "[N;R0]=[N;R0]=[C;R0]",
    "[N+]#N-*",
    "[N;R0](~N)~O",
    "N(~O)(~O)(~O)-*",
    "[N+]([O-])(=C)-*",
    "[!$([C,c]-N(=O)~O);$([!O]~[N;R0]=O)]",
    "N#C[C;R0;X4]O[!$(O=[C,S])]",
    "O=CON1C(=O)CCC1=O",
    "O=COn1cncc1",
    "Fc1c(OC=O)c(F)c(F)c(F)c1F",
    "[NH]=C([NH2])c",
    "O=CN[OH]",
    "[NH;R0][NH;R0]",
    "[$(O=C[CH](C=O)C=O),$(N#C[CH](-C=O)-C=O)]",
    "[!$(O=[C,S])][N;R0]=[C;R0]([C,c])[C,c]",
    "O=CC([Cl,Br,I,F])([Cl,Br,I,F])[Cl,Br,I,F]",
    "[$([NH2]),$([NH][c,CX4]),$(N([c,CX4])[c,CX4])]C(=O)O[$([#6]);!$(C=[O,S,N])]",
    "[$([NH2]),$([NH][c,CX4]),$(N([c,CX4])[c,CX4])]C(=O)[OH]",
    'NC(=O)N',
    'NC(=N)OC',
    'NC(=O)OC',
]

print("常量和SMARTS模式定义完成")

ALKYNE_PATTERN = Chem.MolFromSmarts("C#C")
if ALKYNE_PATTERN is None:
    raise ValueError("ALKYNE_PATTERN SMARTS 编译失败")

UNALLOWED_PATS = []
for s in UNALLOWED_SMARTS:
    pat = Chem.MolFromSmarts(s)
    UNALLOWED_PATS.append(pat)


def is_symmetric(mol, bond_idx) -> bool:
    """
    判断指定的某个三键是否对称。
    使用 CanonicalRankAtoms 判断键两端的原子环境是否一致。
    """
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    bond = mol.GetBondWithIdx(bond_idx)
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    return ranks[idx1] == ranks[idx2]


def check_alkyne_size(mol, bond_idx, max_c) -> bool:
    """
    判断指定三键的某一端，是否满足 “R基团碳数 <= max_c”。
    采用 FragmentOnBonds 虚拟切断法。
    """
    try:
        frags = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
        mol_frags = Chem.GetMolFrags(frags, asMols=True)
    except Exception:
        return False

    if len(mol_frags) != 2:
        return False

    frag1, frag2 = mol_frags
    c1 = sum(1 for a in frag1.GetAtoms() if a.GetSymbol() == 'C')
    c2 = sum(1 for a in frag2.GetAtoms() if a.GetSymbol() == 'C')

    # 每个碎片包含一个炔碳，因此：R碳数 = 碎片碳数 - 1
    # R <= max_c  <=>  碎片碳数 <= max_c + 1
    threshold = max_c + 1
    return (c1 <= threshold) or (c2 <= threshold)


def ok_smiles(smi: str):
    """
    通过则返回标准化 canonical SMILES；不通过返回 None
    """
    if not smi or not isinstance(smi, str):
        return None
    smi = smi.strip()

    # --- Stage 1: 快速字符串过滤 ---
    if "#" not in smi:
        return None
    if "." in smi:
        return None
    if "@" in smi:
        return None

    # --- Stage 2: 解析分子 ---
    try:
        m = Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        return None
    if m is None:
        return None

    Chem.RemoveStereochemistry(m)

    # --- Stage 3: 基本属性过滤 ---
    if MolWt(m) > 300.0:
        return None

    for atom in m.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return None
        # 没有自由电子，不是自由基
        if atom.GetNumRadicalElectrons() != 0:
            return None
        # 不要同位素标记
        if atom.GetIsotope() != 0:
            return None
        if atom.GetFormalCharge() != 0:
            return None

    # 1. 碳原子数在 4~12 之间（3 < 计数 <=12 → 4~12）
    carbon_count = sum(1 for atom in m.GetAtoms() if atom.GetAtomicNum() == 6)
    if not (3 < carbon_count <= 12):
        return None

    # 2. 重原子数在 4~15 之间（重原子=非氢原子，GetNumHeavyAtoms 无需传参）
    heavy_atom_count = m.GetNumHeavyAtoms()  # 修正：删除多余的 m 参数
    if not (3 < heavy_atom_count <= 15):
        return None

    try:
        if not (-1.0 <= MolLogP(m) <= 4.0):
            return None
    except Exception:
        return None

    if len(Chem.GetMolFrags(m)) != 1:
        return None

    # 整体带带电量为 0，不是离子
    if Chem.GetFormalCharge(m) != 0:
        return None

    # --- Stage 4: 结构特征过滤 ---
    if rdMolDescriptors.CalcNumBridgeheadAtoms(m) > 0:
        return None
    if rdMolDescriptors.CalcNumSpiroAtoms(m) > 0:
        return None

    # 环大小过滤
    # 使用RDKit的环信息检测三元环和8元环以上的环
    ring_info = m.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_size = len(ring)
        # 过滤容易开环的3元环和4元环
        if ring_size <= 4:
            return None
        # 过滤大环
        if ring_size >= 8:
            return None

    # --- Stage 5: 核心三键逻辑 ---
    matches = m.GetSubstructMatches(ALKYNE_PATTERN)
    if len(matches) != 1:
        return None

    idx1, idx2 = matches[0]
    bond = m.GetBondBetweenAtoms(idx1, idx2)
    if bond is None:
        return None
    bond_idx = bond.GetIdx()

    for pat in UNALLOWED_PATS:
        if m.HasSubstructMatch(pat):
            return None

    # 小基团筛选：R 基团碳数 <= 2（即甲基/乙基）
    if not check_alkyne_size(m, bond_idx, max_c=2):
        return None

    # 排除对称炔（R1==R2）
    if is_symmetric(m, bond_idx):
        return None

    # --- Stage 6: 返回标准化 SMILES ---
    try:
        can = Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None

    return can


print("核心函数定义完成")


def process_chunk(chunk_data):
    """处理一个数据块的辅助函数，同时处理 cid 和 smiles"""
    results = []
    for cid, smi in chunk_data:
        can = ok_smiles(smi)
        if can:
            results.append(f"{cid}\t{can}\n")
    return results


def main_parallel(
    in_path,
    out_path,
    nproc=None,
    csv_chunksize=100000,
    pool_chunksize=2000,
):
    """
    多进程处理主函数
    """
    if nproc is None:
        nproc = max(1, cpu_count() - 2)

    print(f"开始处理: {in_path}")
    print(f"使用进程数: {nproc}")
    print(f"数据块大小: {csv_chunksize}")

    total_processed = 0
    total_passed = 0

    try:
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = sum(1 for _ in f)
        print(f"文件总行数: {total_lines:,}")
    except Exception:
        total_lines = None

    reader = pd.read_csv(
        in_path,
        header=None,
        names=["cid", "smiles"],
        chunksize=csv_chunksize,
        engine="c",
        on_bad_lines="skip",
        sep="\t",
        dtype={"cid": "string", "smiles": "string"},
    )

    with open(out_path, "wt", encoding="utf-8") as fout, Pool(processes=nproc) as pool:
        pbar = tqdm(total=total_lines, desc="过滤进度", unit="行")

        for chunk_idx, chunk in enumerate(reader):
            valid_data = []
            # 比 iterrows 更快一点
            for cid, smi in zip(chunk["cid"].tolist(), chunk["smiles"].tolist()):
                if pd.isna(cid) or pd.isna(smi):
                    continue
                valid_data.append((str(cid), str(smi)))

            if len(valid_data) < pool_chunksize * 2:
                results = []
                for cid, smi in tqdm(valid_data, desc=f"块 {chunk_idx} (单进程)", leave=False):
                    can = ok_smiles(smi)
                    if can:
                        results.append(f"{cid}\t{can}\n")
            else:
                chunk_size = max(100, len(valid_data) // (nproc * 2))
                chunks = [valid_data[i:i + chunk_size] for i in range(0, len(valid_data), chunk_size)]
                results_lists = list(
                    tqdm(
                        pool.imap(process_chunk, chunks),
                        total=len(chunks),
                        desc=f"块 {chunk_idx} (多进程)",
                        leave=False,
                    )
                )
                results = []
                for rlist in results_lists:
                    results.extend(rlist)

            if results:
                fout.writelines(results)
                total_passed += len(results)

            total_processed += len(valid_data)
            pbar.update(len(chunk))

            if chunk_idx % 10 == 0:
                print(f"\n进度: 处理 {total_processed:,} 行, 通过 {total_passed:,} 个分子")
                print(f"通过率: {(total_passed / total_processed * 100):.2f}%" if total_processed else "通过率: 0%")

        pbar.close()

    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"总共处理: {total_processed:,} 个SMILES")
    print(f"通过过滤: {total_passed:,} 个分子")
    print(f"通过率: {(total_passed / total_processed * 100):.2f}%" if total_processed else "通过率: 0%")
    print(f"结果保存到: {out_path}")
    print("=" * 50)

    return total_processed, total_passed


print("✅ 多进程处理函数定义完成")

params = {
    "in_path": "/mnt/data/kimariyb/dataset/pubchem/CID-SMILES.csv",
    "out_path": "/mnt/data/kimariyb/dataset/pubchem/alkynes_cid_smi.csv",
    "nproc": None,
    "csv_chunksize": 100000,
    "pool_chunksize": 5000,
}

total_processed, total_passed = main_parallel(**params)
