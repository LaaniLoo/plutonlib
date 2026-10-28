#!/usr/bin/env python3

from pathlib import Path 
from platform import release
import subprocess
import re 
from datetime import date

def log_str_split(line, split_str, strip_ticks=False):
    value = line.split(split_str, 1)[1].strip()
    return value.strip("`") if strip_ticks else value

def get_commits_from_log(log_md_path):
    log = Path(log_md_path).expanduser()
    if not log.is_file():
        raise FileNotFoundError(f"Unable to find log file with path: '{log}'")

    text = log.read_text()
    ticked=False
    commits = []
    keys = ["File","Type","Summary","Description"]
    current = None   # which field we're collecting into
    for raw in text.splitlines():
        line = raw.strip()

        if line.startswith("- ["):
            ticked = line.startswith("- [x]")
            current = None
            if ticked:
                commits.append({})
            
        elif line == "---":
            current = None

        elif line.startswith("Scope:") and ticked:
            commits[-1]["Scope"] = log_str_split(line, ":", strip_ticks=True)
        elif ticked and any(line.startswith(k + ":") for k in keys):
            current = line.split(":", 1)[0]
            commits[-1][current] = log_str_split(line, ":")
        elif ticked and current == "Description":
            commits[-1]["Description"] += "\n" + raw.rstrip()

    for i, cm in enumerate(commits, 1):       # same indent as the loop above
        missing = [k for k in ("Scope", "Type", "Summary") if k not in cm]
        if missing:
            raise ValueError(f"ticked entry #{i} is missing: {', '.join(missing)}")

    return commits

def get_tag_version(src_path,log_md_path):
    src = Path(src_path).expanduser()
    log = Path(log_md_path).expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"Unable to find git repo src in '{src}'")
    if not log.is_file():
        raise FileNotFoundError(f"Unable to find log file with path: '{log}'")

    tag = subprocess.check_output(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=src, text=True).strip()

    version = None
    is_newer = True
    text = log.read_text()
    for line in text.splitlines():
        if line.startswith("version:"):
            version = log_str_split(line, ":")
            break
    else:
        raise ValueError(f"no version: line in {log.name}")

    latest = tuple(map(int, tag.lstrip("v").split(".")))

    try:
        new = tuple(map(int, version.split(".")))
    except ValueError as e:
        raise ValueError(f"Invalid version found in {log.name} ({version}), please use format (x.x.x) e.g. '{tag.lstrip('v')}'.\n{e}")
    if len(new) != 3:
        raise ValueError(f"version in {log.name} ({version}) must have 3 parts e.g. '{tag.lstrip('v')}'")

    if new < latest:
        # print(f"{version} is older than most recent git tag: {tag}, updating {log.name}...")
        new_ver =  f"{latest[0]}.{latest[1]}.{latest[2]}"
        log.write_text(text.replace(f"version: {version}", f"version: {new_ver}",1))
        version = new_ver

    release = new > latest #new is value in log
    return version,release

def write_commits_md(commits_md_path,commits,version):
    out_file = Path(commits_md_path).expanduser()
    if not out_file.is_file():
        raise FileNotFoundError(f"Unable to find commits file with path: '{out_file}'")

    groups = {}
    for cm in commits:
        groups.setdefault(cm["Scope"], []).append(cm)
    
    fence = "`" * 3 
    blocks = [f"---\nversion: {version}\n---"]
    for name,entries in groups.items():
        blocks.append(f"#### **{name}**")
        for cm in entries:
            file_name = f"`{cm['File']}`" if cm.get("File") else ""
            subject = f"{cm['Type']}({name}): {cm['Summary']}"
            desc = cm.get("Description", "").split("\n")[0]
            blocks.append(
                f"**File:** {file_name}\n"
                f"**Summary:**\n{fence}bash\n{subject}\n{fence}\n"
                f"**Description:**\n{fence}bash\n{desc}\n{fence}\n"
            )

    cmd = f"git tag v{version} && git push origin v{version}"
    blocks.append(
        f"#### **Tag and push command:**\n{fence}bash\n{cmd}\n{fence}\n"
    )    
    out_file.write_text("\n".join(blocks))

def update_changelog(changelog_path,commits,version):
    cl = Path(changelog_path).expanduser()
    if not cl.is_file():
        raise FileNotFoundError(f"Unable to find CHANGELOG.md file with path: '{cl}'")

    existing = cl.read_text() if cl.is_file() else "# Changelog\n"
    if f"## [{version}]" in existing:
        raise ValueError(f"{version} is already in {cl}")

    groups = {}
    for cm in commits:
        groups.setdefault(cm["Scope"], []).append(cm)

    heading = f"## [{version}] - {date.today():%Y-%m-%d}"
    out = [heading]
    for scope, entries in groups.items():
        out.append(f"#### **{scope}**")
        for cm in entries:                      # pass 1: commit + short desc
            file_name = f" (`{cm['File']}`)" if cm.get("File") else ""
            out.append(f"* ***{cm['Type']}:*** {cm['Summary']}{file_name}")
            first = cm.get("Description", "").split("\n")[0].strip()
            if first:
                out.append(f"   * {first}")
        for cm in entries:                      # pass 2: remaining description
            rest = "\n".join(cm.get("Description", "").split("\n")[1:]).strip()
            if rest:
                label = cm.get("File") or cm["Summary"]
                out.append(f"\n**`{label}`**\n{rest}\n")
        out.append("\n---")

    changelog = re.sub(r"\n{3,}", "\n\n", "\n".join(out))

    idx = existing.find("\n## [")
    if idx == -1:      # no releases yet, so append after the title
        new = existing.rstrip("\n") + "\n\n" + changelog + "\n"
    else:
        new = existing[:idx + 1] + changelog + "\n\n" + existing[idx + 1:]

    # cl.with_suffix(".bak").write_text(existing)
    cl.write_text(new)

def mark_released(log_md_path):
    log = Path(log_md_path).expanduser()
    lines = log.read_text().splitlines(keepends=True)   # fresh read
    lines = [l.replace("[x]", "[r]", 1) if l.lstrip().startswith("- [x]") else l
             for l in lines]
    log.write_text("".join(lines))

def main():
    log_path = "~/Main/alains_obsidian_vault_local/plutonlib/log.md"
    commits_path = "~/Main/alains_obsidian_vault_local/plutonlib/commits.md"
    src_path = "~/plutonlib"
    changelog_path = "~/plutonlib/CHANGELOG.md"

    commits = get_commits_from_log(log_path)
    version,release = get_tag_version(src_path,log_path)
    print("Writing commits to commits.md...")
    write_commits_md(commits_path,commits,version)
    if release:
        print("Updating CHANGELOG.md...")
        update_changelog(changelog_path,commits,version)
        mark_released(log_path)
if __name__ == "__main__":
    main()