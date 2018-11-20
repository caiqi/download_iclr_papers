"""
create a shell scripts for downloading all pdfs to pdfs folder
"""
import json
import os

if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

if not os.path.exists("iclr2019.json"):
    print("First run download.py to make the iclr2019 file")
    exit(-1)

filtered_papers = json.load(open("iclr2019.json", "r", encoding="utf-8"))
all_cmd = []
for m in filtered_papers:
    save_path = m[1].replace(" ", "_") + "_" + m[0]
    save_path = ''.join([i for i in save_path if i.isalpha() or i == "_"])
    all_cmd.append("wget -O pdfs/" + save_path + ".pdf " + m[-1])
with open("download_pdf.sh", "w") as g:
    g.write("set -x\n")
    for m in all_cmd:
        g.write(m.strip() + "\n")
        g.write("sleep 2\n\n")
