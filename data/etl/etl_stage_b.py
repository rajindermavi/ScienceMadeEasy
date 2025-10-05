from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import config

LATEXPAND_BIN = shutil.which("latexpand")
HAVE_LATEXPAND = LATEXPAND_BIN is not None
PANDOC_BIN = shutil.which("pandoc")
HAVE_PANDOC = PANDOC_BIN is not None
DETEX_BIN = shutil.which("detex")
HAVE_DETEX = DETEX_BIN is not None

# Commands that include other TeX sources we want to inline.
_INCLUDE_CMD_RE = re.compile(r"\\(input|include|subfile)\s*\{([^}]+)\}")
# Remove includegraphics blocks (single command possibly with options).
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{[^}]*\}", re.IGNORECASE)

# Environments to drop entirely (figures, tables, tikz pictures / charts).
_DROP_ENVIRONMENTS = (
    "figure",
    "figure*",
    "table",
    "table*",
    "tikzpicture",
    "axis",
)

def _strip_comments(text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""
    stripped_lines: list[str] = []
    for line in text.splitlines():
        i = 0
        while i < len(line):
            if line[i] == "%":
                if i > 0 and line[i - 1] == "\\":
                    # Escaped percent sign; keep it and continue.
                    i += 1
                    continue
                line = line[:i]
                break
            i += 1
        stripped_lines.append(line)
    return "\n".join(stripped_lines)


def _iter_tex_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.tex")):
        if path.is_file():
            yield path


def _score_tex_file(path: Path) -> int:
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return -1
    score = 0
    if "\\documentclass" in sample:
        score += 2
    if "\\begin{document}" in sample:
        score += 3
    if "\\title" in sample:
        score += 1
    if path.name.lower().startswith("main"):
        score += 1
    if path.name.lower().startswith("arxiv"):
        score += 1
    return score


def _find_main_tex(root: Path) -> Path | None:
    candidates = list(_iter_tex_files(root))
    if not candidates:
        return None
    scored = sorted((( _score_tex_file(p), p) for p in candidates), reverse=True)
    best_score, best_path = scored[0]
    if best_score <= 0:
        return None
    return best_path


def _resolve_include(base_dir: Path, target: str, root: Path) -> Path | None:
    target = target.strip()
    if not target:
        return None
    target_path = Path(target)
    search_paths: Iterable[Path]
    if target_path.is_absolute():
        search_paths = [target_path]
    else:
        # Try relative to the current file, then to the paper root.
        rel_candidates = [base_dir / target_path, root / target_path]
        search_paths = rel_candidates
    extensions = ["", ".tex", ".ltx", ".tikz", ".sty"]
    for base in search_paths:
        if base.suffix:
            # User provided an explicit suffix; just test that path.
            if base.exists():
                return base
            continue
        for ext in extensions:
            candidate = base.with_suffix(ext) if ext else base
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _inline_includes(path: Path, root: Path, visited: set[Path]) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    text = _strip_comments(raw)

    def replace(match: re.Match[str]) -> str:
        include_target = match.group(2)
        resolved = _resolve_include(path.parent, include_target, root)
        if not resolved:
            return ""
        resolved = resolved.resolve()
        if resolved in visited:
            return ""
        visited.add(resolved)
        return _inline_includes(resolved, root, visited)

    return _INCLUDE_CMD_RE.sub(replace, text)


def _run_latexpand(main_tex: Path, root: Path) -> str | None:
    if not HAVE_LATEXPAND:
        return None
    try:
        main_arg = str(main_tex.relative_to(root))
    except ValueError:
        main_arg = str(main_tex)
    cmd = [LATEXPAND_BIN, "--empty-comments"]
    bbl_paths = sorted(root.rglob("*.bbl"))
    for bbl in bbl_paths:
        try:
            rel_bbl = str(bbl.relative_to(root))
        except ValueError:
            rel_bbl = str(bbl)
        cmd.extend(["--expand-bbl", rel_bbl])
    cmd.append(main_arg)
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout

def _convert_to_md(latex_file: Path):

    md_ok = False
    warning = ""
    outfile = (config.MD_VERSION_DIR / latex_file.name).with_suffix('.md')

    if not HAVE_PANDOC:
        return md_ok, outfile, 'Pandoc missing. Skipping Markdown'
    
    cmd = [
            "pandoc", "-f", "latex", "-t", "gfm", "--wrap", "none",
            str(latex_file), "-o", str(outfile)
    ]

    try:
        subprocess.run(cmd,check=True)
        md_ok = True
    except subprocess.CalledProcessError as e:
        warning = f"pandoc→Markdown failed ({e.returncode}); skipping Markdown."

    return md_ok, outfile, warning

def _convert_to_txt(latex_file: Path):

    txt_ok = False 
    outfile = (config.TEXT_VERSION_DIR / latex_file.name).with_suffix('.txt')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if HAVE_DETEX:    
        cmd = ["detex", str(latex_file)]
        try:
            res = subprocess.run(cmd, check=True, capture_output=True)
            outfile.write_bytes(res.stdout)
            txt_ok = True
        except subprocess.CalledProcessError as e:
            warnings.append(f"detex failed ({e.returncode}); trying pandoc -t plain.")
    if not txt_ok and HAVE_PANDOC:
        try:
            res = subprocess.run(["pandoc", "-f", "latex", "-t", "plain", str(latex_file)],
                                 check=True, capture_output=True)
            outfile.write_bytes(res.stdout)
            txt_ok = True
        except subprocess.CalledProcessError as e:
            warnings.append(f"pandoc→plain failed ({e.returncode}); using naive stripper.")
    if not txt_ok:
        # Very naive fallback: remove lines starting with % and strip simple commands.
        raw = latex_file.read_text(errors="ignore")
        raw = "\n".join(l for l in raw.splitlines() if not l.lstrip().startswith("%"))
        # Zap \command{...} and \command[...] and \command
        raw = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", raw)
        # Remove TeX comments mid-line
        raw = re.sub(r"(?<!\\)%.*", "", raw)
        # Collapse multiple spaces
        raw = re.sub(r"[ \t]+", " ", raw)
        outfile.write_text(raw)
        warnings.append("Used naive text stripper (install detex and/or pandoc for better output).")
        txt_ok = True

    return txt_ok, outfile, warnings
    


def _strip_environments(text: str, environments: Iterable[str]) -> str:
    env_list = list(environments)
    if not env_list:
        return text
    env_pattern = "|".join(re.escape(env) for env in env_list)
    begin_re = re.compile(r"\\begin\{(" + env_pattern + r")\}(?:\[[^\]]*\])?", re.IGNORECASE)

    result: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        begin_match = begin_re.search(text, index)
        if not begin_match:
            result.append(text[index:])
            break
        env_name = begin_match.group(1)
        start = begin_match.start()
        result.append(text[index:start])
        # Find matching \end{env_name}
        end_re = re.compile(r"\\end\{" + re.escape(env_name) + r"\}", re.IGNORECASE)
        scan_pos = begin_match.end()
        depth = 1
        while depth > 0:
            next_begin = begin_re.search(text, scan_pos)
            next_end = end_re.search(text, scan_pos)
            if not next_end:
                # Unmatched begin; abort and keep the rest as-is
                result.append(text[start:])
                return "".join(result)
            if next_begin and next_begin.start() < next_end.start() and next_begin.group(1).lower() == env_name.lower():
                depth += 1
                scan_pos = next_begin.end()
                continue
            depth -= 1
            scan_pos = next_end.end()
        index = scan_pos
    return "".join(result)


def _clean_text(text: str) -> str:
    without_envs = _strip_environments(text, _DROP_ENVIRONMENTS)
    without_figures = _INCLUDEGRAPHICS_RE.sub("", without_envs)
    return without_figures


def prepare_latex_corpus(source_root: Path | None = None, output_root: Path | None = None) -> list[Path]:
    source_root = source_root or config.TAR_EXTRACT_DIR
    output_root = output_root or config.LATEX_FILTER_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for paper_dir in sorted(source_root.iterdir()):
        if not paper_dir.is_dir():
            continue
        main_tex = _find_main_tex(paper_dir)
        if main_tex is None:
            # Fallback: concatenate all .tex files in alpha order.
            pieces = []
            for tex_file in _iter_tex_files(paper_dir):
                try:
                    file_text = tex_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                pieces.append(_strip_comments(file_text))
            combined = "\n\n".join(pieces)
        else:
            expanded = _run_latexpand(main_tex, paper_dir)
            if expanded is None:
                visited = {main_tex.resolve()}
                combined = _inline_includes(main_tex, paper_dir, visited)
            else:
                combined = expanded
        cleaned = _clean_text(combined)
        cleaned = cleaned.strip()
        output_path = output_root / f"{paper_dir.name}.tex"
        output_path.write_text(cleaned + "\n", encoding="utf-8")
        written.append(output_path)
    return written

def latex_conversion(files):
    
    md_conversions = []
    txt_convsersions = []

    for file in files:

        md_ok, md_file, md_warning = _convert_to_md(file)
        if md_ok:
            md_conversions.append(md_file)
        else:
            print(md_warning)
        txt_ok, txt_file, txt_warnings = _convert_to_txt(file)
        if txt_ok:
            txt_convsersions.append(txt_file)
        else:
            print(txt_warnings)

    return md_conversions, txt_convsersions

if __name__ == "__main__":
    written_paths = prepare_latex_corpus()
    md_paths, txt_paths = latex_conversion(written_paths)
    for path in written_paths:
        print(f"Wrote {path}")
