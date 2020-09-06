# Copyright (c) 2017 by David Cournapeau
# This software is released under the MIT license. See LICENSE file for the
# actual license.
import ast
import os.path
import re
import subprocess


_DOT_NUMBERS_RE = re.compile("v?(\d+!)?(\d+(\.\d+)*)")

_R_RC = re.compile("rc(\d+)$")

_TEMPLATE = """\
# THIS FILE IS GENERATED FROM {package_name} SETUP.PY
version = '{final_version}'
full_version = '{full_version}'
git_revision = '{git_revision}'
is_released = {is_released}

version_info = {version_info}
"""


def _is_rc(version):
    return _R_RC.search(version) is not None


def _rc_number(version):
    m = _R_RC.search(version)
    assert m is not None, version
    return int(m.groups()[0])


class _AssignmentParser(ast.NodeVisitor):
    """ Simple parser for python assignments."""
    def __init__(self):
        self._data = {}

    def parse(self, s):
        self._data.clear()

        root = ast.parse(s)
        self.visit(root)
        return self._data

    def generic_visit(self, node):
        if type(node) != ast.Module:
            raise ValueError(
                "Unexpected expression @ line {0}".format(node.lineno),
                node.lineno
            )
        super(_AssignmentParser, self).generic_visit(node)

    def visit_Assign(self, node):
        value = ast.literal_eval(node.value)
        for target in node.targets:
            self._data[target.id] = value


def parse_version(path):
    with open(path, "rt") as fp:
        return _AssignmentParser().parse(fp.read())["version"]


# Return the git revision as a string
def git_version(since_commit=None):
    """
    Compute the current git revision, and the number of commits since
    <since_commit>.

    Parameters
    ----------
    since_commit : string or None
        If specified and not None, git_count will be the number of commits
        between this value and HEAD. Useful to e.g. compute a build number.
    """
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    if since_commit is None:
        since_commit = "HEAD"
    else:
        since_commit += ".."

    try:
        out = subprocess.check_output(
            ['git', 'rev-list', '--count', since_commit]
        )
        git_count = int(out.strip().decode('ascii'))
    except OSError:
        git_count = 0

    return git_revision, git_count


def write_version_py(package_name, version, since_commit=None,
                     is_released=False, filename=None):
    if filename is None:
        filename = os.path.abspath(os.path.join(package_name, "_version.py"))

    m = _DOT_NUMBERS_RE.search(version)
    if m is None:
        raise ValueError("Format not supported: {!r}".format(version))

    if not os.path.exists(".git") and os.path.exists(filename):
        # must be a source distribution, use existing version file
        return parse_version(filename)

    if os.path.exists('.git'):
        git_rev, build_number = git_version(since_commit)
    else:
        git_rev, build_number = "Unknown", 0

    if _is_rc(version):
        release_level = "rc"
    elif not is_released:
        release_level = "dev"
    else:
        release_level = "final"

    dot_numbers_string = m.groups()[1]
    full_version = dot_numbers_string

    if is_released:
        final_version = full_version
        if _is_rc(version):
            serial = _rc_number(version)
        else:
            serial = 0
    else:
        full_version += '.dev' + str(build_number)
        final_version = full_version
        if _is_rc(version):
            serial = _rc_number(version)
        else:
            serial = build_number

    dot_numbers = tuple(int(item) for item in dot_numbers_string.split("."))
    version_info = dot_numbers + (release_level, serial)

    with open(filename, "wt") as fp:
        data = _TEMPLATE.format(
            final_version=final_version, full_version=full_version,
            git_revision=git_rev, is_released=is_released,
            version_info=version_info, package_name=package_name.upper(),
        )
        fp.write(data)

    return full_version
