from Bio import Entrez, SeqIO

# Account Info
Entrez.email = "kid1064@gmail.com"

handle = Entrez.esearch(
    db="nucleotide",
    term='CRT[Gene Name] AND "Plasmodium falciparum"[Organism]'
)

rec_list = Entrez.read(handle)

if rec_list['RetMax'] < rec_list['Count']:
    handle = Entrez.esearch(
        db="nucleotide",
        term='CRT[Gene Name] AND "Plasmodium falciparum"[Organism]',
        retmax=rec_list["Count"]
    )
    rec_list = Entrez.read(handle)

id_list = rec_list["IdList"]

hdl = Entrez.efetch(
    db="nucleotide",
    id=id_list,
    rettype="gb"
)

recs = list(SeqIO.parse(hdl, "gb"))

for rec in recs:
    if rec.name == "KM288867":
        break
print(rec.name)
print(rec.description)