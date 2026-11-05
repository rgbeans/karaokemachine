"""Neural karaoke-map autogenerator for KaraokeMachine.

This package is intentionally independent from the Rust application. It reads
the local song library, trains only on maps listed in ``approved.json``, and
writes draft map files that the app can play and grade.
"""

__version__ = "0.1.0"
