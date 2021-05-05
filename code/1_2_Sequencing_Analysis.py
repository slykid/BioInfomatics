from Bio import Entrez, SeqIO

Entrez.email="kid1064@gmail.com"

handle = Entrez.efetch(
    db="nucleotide"
    , id=["NM_002299"]  # 락타아제 유전자
    , rettype="fasta"
)
seq = SeqIO.read(handle, "fasta")

# 락타아제 유전자 서열 데이터 저장
w_hdl = open("data/lactase.fasta", "w")
w_seq = seq[11:5795]
SeqIO.write([w_seq], w_hdl, 'fasta')
w_hdl.close()

# 락타아제 유전자 서열 데이터 로드
recs = SeqIO.parse("data/lactase.fasta", "fasta")
for rec in recs:
    seq = rec.seq
    print(rec.description)
    print(seq[:10])
    print(seq.alphabet)