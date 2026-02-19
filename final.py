from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any


# =========================
# Token + Error
# =========================
@dataclass
class Token:
    type: str     # KW, IDENT, INT, FLOAT, STRING, OP, SEP, EOF
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"{self.type}({self.value!r})@{self.line}:{self.col}"


class LexerError(Exception):
    pass


# =========================
# Lexer
# =========================
class Lexer:
    """
    Lexical Analyzer:
    - removes whitespace/comments
    - outputs tokens
    - builds a simple lexical table (identifiers + constants)
    """

    KEYWORDS: Set[str] = {
        "let", "const", "if", "else", "while", "for",
        "fn", "return", "print", "true", "false", "null"
    }

    # Operators (longest first)
    OPERATORS = [
        "++", "--", "->",                 # added (from common symbol sets + examples like i++)
        "===", "==", "!=", "<=", ">=",
        "+=", "-=", "*=", "/=",
        "&&", "||",
        "=", "+", "-", "*", "/", "%", "<", ">", "!"
    ]

    # Symbols / separators (punctuation)
    SEPARATORS: Set[str] = {"(", ")", "{", "}", "[", "]", ",", ";", ":", "."}

    # Comments
    LINE_COMMENT = "//"
    BLOCK_COMMENT_START = "/*"
    BLOCK_COMMENT_END = "*/"

    def __init__(self, source: str):
        self.source = source
        self.i = 0
        self.line = 1
        self.col = 1

        # Simple "Lexical Table" (symbol tables for later compiler phases)
        self.identifiers: Set[str] = set()
        self.constants: Set[str] = set()

    # ---------- helpers ----------
    def peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.source[j] if j < len(self.source) else "\0"

    def advance(self) -> str:
        ch = self.peek()
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def startswith(self, s: str) -> bool:
        return all(self.peek(k) == s[k] for k in range(len(s)))

    def token(self, ttype: str, value: str, line: int, col: int) -> Token:
        return Token(ttype, value, line, col)

    # ---------- lexing ----------
    def lex_identifier_or_keyword(self) -> Token:
        line, col = self.line, self.col
        s = ""
        while self.peek().isalnum() or self.peek() == "_":
            s += self.advance()

        if s in self.KEYWORDS:
            return self.token("KW", s, line, col)

        self.identifiers.add(s)  # lexical table entry
        return self.token("IDENT", s, line, col)

    def lex_number(self) -> Token:
        """
        Supports:
        - INT: 123
        - FLOAT: 12.34
        - SCI: 5.1e3, 1e-2, 2.0E+5
        """
        line, col = self.line, self.col
        s = ""

        # integer part
        while self.peek().isdigit():
            s += self.advance()

        is_float = False

        # fractional part
        if self.peek() == "." and self.peek(1).isdigit():
            is_float = True
            s += self.advance()  # '.'
            while self.peek().isdigit():
                s += self.advance()

        # scientific notation part
        if self.peek() in ("e", "E"):
            # only treat as sci-notation if followed by digit or sign+digit
            nxt = self.peek(1)
            nxt2 = self.peek(2)
            if nxt.isdigit() or (nxt in "+-" and nxt2.isdigit()):
                is_float = True
                s += self.advance()  # e/E
                if self.peek() in "+-":
                    s += self.advance()
                while self.peek().isdigit():
                    s += self.advance()

        ttype = "FLOAT" if is_float else "INT"
        self.constants.add(s)  # lexical table entry
        return self.token(ttype, s, line, col)

    def lex_string(self) -> Token:
        """
        Supports:
        - 'single' and "double"
        - '' inside single quotes becomes literal '
        - escapes: \n, \t, \\, \', \"
        """
        quote = self.peek()
        line, col = self.line, self.col
        self.advance()  # opening quote

        out = ""
        while True:
            ch = self.peek()

            if ch == "\0":
                raise LexerError(f"Unterminated string at {line}:{col}")

            if ch == quote:
                if quote == "'" and self.peek(1) == "'":  # '' => literal '
                    self.advance(); self.advance()
                    out += "'"
                    continue
                self.advance()  # closing quote
                break

            if ch == "\\":
                self.advance()
                esc = self.peek()
                escapes = {"n": "\n", "t": "\t", "\\": "\\", "'": "'", '"': '"'}
                if esc in escapes:
                    out += escapes[esc]
                    self.advance()
                else:
                    out += "\\" + self.advance()
                continue

            out += self.advance()

        self.constants.add(out)  # lexical table entry (string literal)
        return self.token("STRING", out, line, col)

    def lex_operator(self) -> Optional[Token]:
        line, col = self.line, self.col
        for op in self.OPERATORS:
            if all(self.peek(k) == op[k] for k in range(len(op))):
                for _ in range(len(op)):
                    self.advance()
                return self.token("OP", op, line, col)
        return None

    def skip_comment_if_present(self) -> bool:
        if self.startswith(self.LINE_COMMENT):
            while self.peek() not in ("\n", "\0"):
                self.advance()
            return True

        if self.startswith(self.BLOCK_COMMENT_START):
            self.advance(); self.advance()  # consume /*
            while True:
                if self.peek() == "\0":
                    raise LexerError(f"Unterminated block comment at {self.line}:{self.col}")
                if self.startswith(self.BLOCK_COMMENT_END):
                    self.advance(); self.advance()  # consume */
                    return True
                self.advance()
        return False

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.peek() != "\0":
            ch = self.peek()

            if ch.isspace():
                self.advance()
                continue

            # strings first
            if ch in ("'", '"'):
                tokens.append(self.lex_string())
                continue

            # comments
            if self.skip_comment_if_present():
                continue

            # numbers
            if ch.isdigit():
                tokens.append(self.lex_number())
                continue

            # identifiers/keywords
            if ch.isalpha() or ch == "_":
                tokens.append(self.lex_identifier_or_keyword())
                continue

            # separators
            if ch in self.SEPARATORS:
                line, col = self.line, self.col
                tokens.append(self.token("SEP", self.advance(), line, col))
                continue

            # operators
            op = self.lex_operator()
            if op is not None:
                tokens.append(op)
                continue

            raise LexerError(f"Unexpected character {ch!r} at {self.line}:{self.col}")

        tokens.append(self.token("EOF", "", self.line, self.col))
        return tokens

    def lexical_table(self) -> Dict[str, Any]:
        """
        Returns a simple lexical table (for your report / later phases).
        """
        return {
            "identifiers": sorted(self.identifiers),
            "constants": sorted(self.constants),
            "keywords": sorted(self.KEYWORDS),
            "operators": self.OPERATORS[:],
            "separators": sorted(self.SEPARATORS),
        }


# =========================
# Example run (matches the PDF-style example cases)
# =========================
if __name__ == "__main__":
    code = r"""
    for (i = 1; i < 5.1e3; i++) {
        let msg = "He said 'hi' and I can't leave // not a comment";
        print(msg);
    }
    """

    lexer = Lexer(code)
    toks = lexer.tokenize()
    for t in toks:
        print(t)

    print("\n--- Lexical Table ---")
    print("Identifiers:", lexer.lexical_table()["identifiers"])
    print("Constants:", lexer.lexical_table()["constants"])
