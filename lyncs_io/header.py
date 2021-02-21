"""
Header class, a dictionary that contains information about the file content
"""


class Header(dict):
    """
    Header class, a dictionary that contains information about the file content
    """

    def is_proper(self):
        "Whether it is an header created by lyncs_io"
        return "_lyncs_io" in self
