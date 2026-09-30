"""Inject results into the report: python report/build.py results.json report.html [nanochat_results.json] [nanochat_bytes_results.json] [step_curves.json]"""
import sys
from pathlib import Path

tpl = (Path(__file__).parent / "template.html").read_text()
data = Path(sys.argv[1]).read_text().strip().replace("</", "<\\/")
nc = Path(sys.argv[3]).read_text().strip().replace("</", "<\\/") if len(sys.argv) > 3 else ""
ncb = Path(sys.argv[4]).read_text().strip().replace("</", "<\\/") if len(sys.argv) > 4 else ""
st = Path(sys.argv[5]).read_text().strip().replace("</", "<\\/") if len(sys.argv) > 5 else ""
Path(sys.argv[2]).write_text(tpl.replace("/*DATA*/", data).replace("/*NANOCHAT*/", nc).replace("/*NANOCHAT_BYTES*/", ncb).replace("/*STEPS*/", st))
print(f"wrote {sys.argv[2]}")
