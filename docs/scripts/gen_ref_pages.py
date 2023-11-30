"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent.parent.parent / "libmg"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        continue
        # uncomment for pages for init modules
        # parts = parts[:-1]
        # doc_path = doc_path.with_name("index.md")
        # full_doc_path = full_doc_path.with_name("index.md")
    if len(parts) == 0 or parts[-1] == "__main__" or parts[0] == "tests":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: libmg." + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    lines = filter(lambda x: 'test' not in x, nav.build_literate_nav())
    nav_file.writelines(lines)