from Bio import Entrez, SeqIO

# Account Info
Entrez.email = "kid1064@gmail.com"

# Nucleotide gene-info load
handle = Entrez.esearch(
    db="nucleotide",
    term='CRT[Gene Name] AND "Plasmodium falciparum"[Organism]'
)

# Data load
rec_list = Entrez.read(handle)

# Data transform
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

for feature in rec.features:
    if feature.type == 'gene':
        print(feature.qualifiers['gene'])
    elif feature.type == 'exon':
        loc = feature.location
        print(loc.start, loc.end, loc.strand)
    else:
        print(f"not processed:\n{feature}")

for name, value in rec.annotations.items():
    print(f"{name}-{value}")

print(len(rec.seq))