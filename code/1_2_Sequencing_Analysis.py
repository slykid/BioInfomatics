from Bio import Entrez, SeqIO, Seq
from Bio.Alphabet import IUPAC   # 1.78 이상부터는 제공하지 않음

Entrez.email = "kid1064@gmail.com"

handle = Entrez.efetch(
    db='nucleotide'
    , id=['NM_002299']
    , rettype='fasta'
)
seq = SeqIO.read(handle, "fasta")

# 락타아제 유전자 서열 데이터 저장
w_hdl = open("data/lactase.fasta", "w")
w_seq = seq[11:5795]
w_seq

SeqIO.write([w_seq], w_hdl, 'fasta')
w_hdl.close()

# 락타아제 유전자 서열 데이터 로드
recs = SeqIO.parse("data/lactase.fasta", "fasta")

for rec in recs:
    seq = rec.seq
    print(rec.description)
    print(seq[:10])
    print(seq.alphabet)  # biopython <= 1.77 에서만 사용가능 / 1.78 이상부터는 제공하지 않음

# 서열 알파벳 변환
seq = Seq.Seq(str(seq), IUPAC.unambiguous_dna)
rna = seq.transcribe()
print(rna)

prot = seq.translate()
print(prot)