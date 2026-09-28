"""Setuptools entry point for the profile-validated native extension."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class HexadecaBuildExt(build_ext):
    """Locate MSVC tools before building when Windows PATH is uninitialized."""

    def build_extensions(self) -> None:
        """Make the discovered MSVC host tools available to setuptools."""

        if os.name == "nt" and shutil.which("cl.exe") is None:
            self._prepend_msvc_tools()
        super().build_extensions()

    @staticmethod
    def _prepend_msvc_tools() -> None:
        """Discover the latest installed x64 MSVC tools with vswhere."""

        program_files_x86 = Path(
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        )
        vswhere = (
            program_files_x86 / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        )
        if not vswhere.is_file():
            raise RuntimeError(
                "MSVC C++20 tools were not found. Install Visual Studio Build Tools "
                "with the Desktop C++ workload."
            )
        installation_path = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
        tool_root = Path(installation_path) / "VC" / "Tools" / "MSVC"
        compiler_directories = sorted(tool_root.glob("*/bin/Hostx64/x64"), reverse=True)
        if not compiler_directories:
            raise RuntimeError(
                "MSVC C++20 tools were not found. Install Visual Studio Build Tools "
                "with the Desktop C++ workload."
            )
        compiler_directory = compiler_directories[0]
        os.environ["PATH"] = os.pathsep.join(
            (str(compiler_directory), os.environ.get("PATH", ""))
        )


setup(
    cmdclass={"build_ext": HexadecaBuildExt},
    ext_modules=[
        Extension(
            "native._hexadeca_native",
            sources=["cpp/hexadeca_native.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=(
                ["/std:c++20", "/O2", "/EHsc"]
                if os.name == "nt"
                else ["-std=c++20", "-O3"]
            ),
            extra_link_args=["/MANIFEST:NO"] if os.name == "nt" else [],
        )
    ],
)
