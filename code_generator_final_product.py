from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Tuple
import argparse
import contextlib
import io
import json


# =========================
# Token + Error
# =========================
@dataclass
class Token:
    type: str
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
    KEYWORDS: Set[str] = {
        "let", "const", "if", "else", "while", "for",
        "fn", "return", "print", "true", "false", "null"
    }

    OPERATORS = [
        "++", "--", "->",
        "===", "==", "!=", "<=", ">=",
        "+=", "-=", "*=", "/=",
        "&&", "||",
        "=", "+", "-", "*", "/", "%", "<", ">", "!"
    ]

    SEPARATORS: Set[str] = {"(", ")", "{", "}", "[", "]", ",", ";", ":", "."}
    LINE_COMMENT = "//"
    BLOCK_COMMENT_START = "/*"
    BLOCK_COMMENT_END = "*/"

    def __init__(self, source: str):
        self.source = source
        self.i = 0
        self.line = 1
        self.col = 1
        self.identifiers: Set[str] = set()
        self.constants: Set[str] = set()

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

    def lex_identifier_or_keyword(self) -> Token:
        line, col = self.line, self.col
        s = ""
        while self.peek().isalnum() or self.peek() == "_":
            s += self.advance()

        if s in self.KEYWORDS:
            return self.token("KW", s, line, col)

        self.identifiers.add(s)
        return self.token("IDENT", s, line, col)

    def lex_number(self) -> Token:
        line, col = self.line, self.col
        s = ""
        while self.peek().isdigit():
            s += self.advance()

        is_float = False
        if self.peek() == "." and self.peek(1).isdigit():
            is_float = True
            s += self.advance()
            while self.peek().isdigit():
                s += self.advance()

        if self.peek() in ("e", "E"):
            nxt = self.peek(1)
            nxt2 = self.peek(2)
            if nxt.isdigit() or (nxt in "+-" and nxt2.isdigit()):
                is_float = True
                s += self.advance()
                if self.peek() in "+-":
                    s += self.advance()
                while self.peek().isdigit():
                    s += self.advance()

        self.constants.add(s)
        return self.token("FLOAT" if is_float else "INT", s, line, col)

    def lex_string(self) -> Token:
        quote = self.peek()
        line, col = self.line, self.col
        self.advance()
        out = ""

        while True:
            ch = self.peek()
            if ch == "\0":
                raise LexerError(f"Unterminated string at {line}:{col}")
            if ch == quote:
                if quote == "'" and self.peek(1) == "'":
                    self.advance(); self.advance()
                    out += "'"
                    continue
                self.advance()
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

        self.constants.add(out)
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
            self.advance(); self.advance()
            while True:
                if self.peek() == "\0":
                    raise LexerError(f"Unterminated block comment at {self.line}:{self.col}")
                if self.startswith(self.BLOCK_COMMENT_END):
                    self.advance(); self.advance()
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
            if ch in ("'", '"'):
                tokens.append(self.lex_string())
                continue
            if self.skip_comment_if_present():
                continue
            if ch.isdigit():
                tokens.append(self.lex_number())
                continue
            if ch.isalpha() or ch == "_":
                tokens.append(self.lex_identifier_or_keyword())
                continue
            if ch in self.SEPARATORS:
                line, col = self.line, self.col
                tokens.append(self.token("SEP", self.advance(), line, col))
                continue
            op = self.lex_operator()
            if op is not None:
                tokens.append(op)
                continue
            raise LexerError(f"Unexpected character {ch!r} at {self.line}:{self.col}")

        tokens.append(self.token("EOF", "", self.line, self.col))
        return tokens

    def lexical_table(self) -> Dict[str, Any]:
        return {
            "identifiers": sorted(self.identifiers),
            "constants": sorted(self.constants),
            "keywords": sorted(self.KEYWORDS),
            "operators": self.OPERATORS[:],
            "separators": sorted(self.SEPARATORS),
        }


# =========================
# Parser Error + AST
# =========================
class ParserError(Exception):
    pass


@dataclass
class ASTNode:
    kind: str
    value: Any = None
    children: List["ASTNode"] = field(default_factory=list)

    def pretty(self, level: int = 0) -> str:
        indent = "  " * level
        s = f"{indent}{self.kind}"
        if self.value is not None:
            s += f": {self.value}"
        s += "\n"
        for child in self.children:
            s += child.pretty(level + 1)
        return s


# =========================
# Parser / Syntax Analyzer
# =========================
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    def current(self) -> Token:
        return self.tokens[self.i]

    def peek(self, k: int = 1) -> Token:
        j = self.i + k
        return self.tokens[j] if j < len(self.tokens) else self.tokens[-1]

    def advance(self) -> Token:
        tok = self.current()
        if self.i < len(self.tokens) - 1:
            self.i += 1
        return tok

    def match(self, ttype: str, value: Optional[str] = None) -> bool:
        tok = self.current()
        if tok.type != ttype:
            return False
        if value is not None and tok.value != value:
            return False
        self.advance()
        return True

    def expect(self, ttype: str, value: Optional[str] = None) -> Token:
        tok = self.current()
        if tok.type != ttype or (value is not None and tok.value != value):
            expected = f"{ttype}" if value is None else f"{ttype}('{value}')"
            raise ParserError(
                f"Syntax error at line {tok.line}, col {tok.col}: "
                f"expected {expected}, got {tok.type}('{tok.value}')"
            )
        self.advance()
        return tok

    def parse(self) -> ASTNode:
        program = ASTNode("Program")
        while self.current().type != "EOF":
            program.children.append(self.statement())
        return program

    def statement(self) -> ASTNode:
        tok = self.current()
        if tok.type == "KW":
            if tok.value in ("let", "const"):
                return self.variable_declaration()
            if tok.value == "print":
                return self.print_statement()
            if tok.value == "if":
                return self.if_statement()
            if tok.value == "while":
                return self.while_statement()
            if tok.value == "for":
                return self.for_statement()
            if tok.value == "fn":
                return self.function_declaration()
            if tok.value == "return":
                return self.return_statement()

        if tok.type == "SEP" and tok.value == "{":
            return self.block()

        expr = self.expression()
        self.expect("SEP", ";")
        return ASTNode("ExpressionStatement", children=[expr])

    def block(self) -> ASTNode:
        self.expect("SEP", "{")
        node = ASTNode("Block")
        while not (self.current().type == "SEP" and self.current().value == "}"):
            if self.current().type == "EOF":
                tok = self.current()
                raise ParserError(f"Unclosed block at line {tok.line}, col {tok.col}")
            node.children.append(self.statement())
        self.expect("SEP", "}")
        return node

    def variable_declaration(self) -> ASTNode:
        kind = self.expect("KW").value
        ident = self.expect("IDENT").value
        init = None
        if self.match("OP", "="):
            init = self.expression()
        self.expect("SEP", ";")
        node = ASTNode("VarDecl", value=kind)
        node.children.append(ASTNode("Identifier", value=ident))
        if init is not None:
            node.children.append(init)
        return node

    def print_statement(self) -> ASTNode:
        self.expect("KW", "print")
        self.expect("SEP", "(")
        expr = self.expression()
        self.expect("SEP", ")")
        self.expect("SEP", ";")
        return ASTNode("Print", children=[expr])

    def if_statement(self) -> ASTNode:
        self.expect("KW", "if")
        self.expect("SEP", "(")
        condition = self.expression()
        self.expect("SEP", ")")
        then_branch = self.statement()
        node = ASTNode("If")
        node.children.extend([condition, then_branch])
        if self.match("KW", "else"):
            node.children.append(self.statement())
        return node

    def while_statement(self) -> ASTNode:
        self.expect("KW", "while")
        self.expect("SEP", "(")
        condition = self.expression()
        self.expect("SEP", ")")
        body = self.statement()
        return ASTNode("While", children=[condition, body])

    def for_statement(self) -> ASTNode:
        self.expect("KW", "for")
        self.expect("SEP", "(")

        init = None
        if not (self.current().type == "SEP" and self.current().value == ";"):
            if self.current().type == "KW" and self.current().value in ("let", "const"):
                init = self.for_var_decl()
            else:
                init = self.expression()
                self.expect("SEP", ";")
        else:
            self.expect("SEP", ";")

        condition = None
        if not (self.current().type == "SEP" and self.current().value == ";"):
            condition = self.expression()
        self.expect("SEP", ";")

        update = None
        if not (self.current().type == "SEP" and self.current().value == ")"):
            update = self.expression()
        self.expect("SEP", ")")

        body = self.statement()
        node = ASTNode("For")
        node.children.append(init if init else ASTNode("EmptyInit"))
        node.children.append(condition if condition else ASTNode("EmptyCondition"))
        node.children.append(update if update else ASTNode("EmptyUpdate"))
        node.children.append(body)
        return node

    def for_var_decl(self) -> ASTNode:
        kind = self.expect("KW").value
        ident = self.expect("IDENT").value
        init = None
        if self.match("OP", "="):
            init = self.expression()
        self.expect("SEP", ";")
        node = ASTNode("VarDecl", value=kind)
        node.children.append(ASTNode("Identifier", value=ident))
        if init is not None:
            node.children.append(init)
        return node

    def function_declaration(self) -> ASTNode:
        self.expect("KW", "fn")
        name = self.expect("IDENT").value
        self.expect("SEP", "(")
        params = ASTNode("Parameters")
        if not (self.current().type == "SEP" and self.current().value == ")"):
            params.children.append(ASTNode("Identifier", value=self.expect("IDENT").value))
            while self.match("SEP", ","):
                params.children.append(ASTNode("Identifier", value=self.expect("IDENT").value))
        self.expect("SEP", ")")
        body = self.block()
        node = ASTNode("FunctionDecl", value=name)
        node.children.extend([params, body])
        return node

    def return_statement(self) -> ASTNode:
        self.expect("KW", "return")
        if self.current().type == "SEP" and self.current().value == ";":
            self.expect("SEP", ";")
            return ASTNode("Return")
        expr = self.expression()
        self.expect("SEP", ";")
        return ASTNode("Return", children=[expr])

    def expression(self) -> ASTNode:
        return self.assignment()

    def assignment(self) -> ASTNode:
        left = self.logical_or()
        if self.current().type == "OP" and self.current().value in ("=", "+=", "-=", "*=", "/="):
            op = self.advance().value
            right = self.assignment()
            return ASTNode("Assign", value=op, children=[left, right])
        return left

    def logical_or(self) -> ASTNode:
        node = self.logical_and()
        while self.match("OP", "||"):
            right = self.logical_and()
            node = ASTNode("BinaryOp", value="||", children=[node, right])
        return node

    def logical_and(self) -> ASTNode:
        node = self.equality()
        while self.match("OP", "&&"):
            right = self.equality()
            node = ASTNode("BinaryOp", value="&&", children=[node, right])
        return node

    def equality(self) -> ASTNode:
        node = self.comparison()
        while self.current().type == "OP" and self.current().value in ("==", "!=", "==="):
            op = self.advance().value
            right = self.comparison()
            node = ASTNode("BinaryOp", value=op, children=[node, right])
        return node

    def comparison(self) -> ASTNode:
        node = self.term()
        while self.current().type == "OP" and self.current().value in ("<", ">", "<=", ">="):
            op = self.advance().value
            right = self.term()
            node = ASTNode("BinaryOp", value=op, children=[node, right])
        return node

    def term(self) -> ASTNode:
        node = self.factor()
        while self.current().type == "OP" and self.current().value in ("+", "-"):
            op = self.advance().value
            right = self.factor()
            node = ASTNode("BinaryOp", value=op, children=[node, right])
        return node

    def factor(self) -> ASTNode:
        node = self.unary()
        while self.current().type == "OP" and self.current().value in ("*", "/", "%"):
            op = self.advance().value
            right = self.unary()
            node = ASTNode("BinaryOp", value=op, children=[node, right])
        return node

    def unary(self) -> ASTNode:
        if self.current().type == "OP" and self.current().value in ("!", "-", "++", "--"):
            op = self.advance().value
            expr = self.unary()
            return ASTNode("UnaryOp", value=op, children=[expr])
        return self.postfix()

    def postfix(self) -> ASTNode:
        node = self.primary()
        while self.current().type == "OP" and self.current().value in ("++", "--"):
            op = self.advance().value
            node = ASTNode("PostfixOp", value=op, children=[node])
        return node

    def primary(self) -> ASTNode:
        tok = self.current()
        if tok.type in ("INT", "FLOAT", "STRING"):
            self.advance()
            return ASTNode("Literal", value=tok.value)

        if tok.type == "KW" and tok.value in ("true", "false", "null"):
            self.advance()
            return ASTNode("Literal", value=tok.value)

        if tok.type == "IDENT":
            self.advance()
            node = ASTNode("Identifier", value=tok.value)
            if self.match("SEP", "("):
                call = ASTNode("Call", children=[node])
                args = ASTNode("Arguments")
                if not (self.current().type == "SEP" and self.current().value == ")"):
                    args.children.append(self.expression())
                    while self.match("SEP", ","):
                        args.children.append(self.expression())
                self.expect("SEP", ")")
                call.children.append(args)
                return call
            return node

        if self.match("SEP", "("):
            expr = self.expression()
            self.expect("SEP", ")")
            return expr

        raise ParserError(
            f"Syntax error at line {tok.line}, col {tok.col}: "
            f"unexpected token {tok.type}('{tok.value}')"
        )


# =========================
# Semantic Analysis
# =========================
class SemanticError(Exception):
    pass


@dataclass
class Symbol:
    name: str
    var_type: str
    kind: str
    initialized: bool = False
    params: Optional[List[str]] = None


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}

    def declare(self, symbol: Symbol):
        if symbol.name in self.symbols:
            raise SemanticError(f"Semantic error: '{symbol.name}' is already declared in this scope")
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        scope = self
        while scope is not None:
            if name in scope.symbols:
                return scope.symbols[name]
            scope = scope.parent
        return None


class SemanticAnalyzer:
    def __init__(self):
        self.global_scope = Scope()
        self.current_scope = self.global_scope
        self.in_function = False

    def enter_scope(self):
        self.current_scope = Scope(self.current_scope)

    def exit_scope(self):
        if self.current_scope.parent is not None:
            self.current_scope = self.current_scope.parent

    def analyze(self, node: ASTNode):
        method_name = f"visit_{node.kind}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)

    def generic_visit(self, node: ASTNode):
        for child in node.children:
            self.analyze(child)

    def get_literal_type(self, value: Any) -> str:
        if value in ("true", "false"):
            return "bool"
        if value == "null":
            return "null"
        if isinstance(value, str):
            try:
                int(value)
                return "int"
            except Exception:
                pass
            try:
                float(value)
                if "." in value or "e" in value.lower():
                    return "float"
            except Exception:
                pass
            return "string"
        return "unknown"

    def is_numeric(self, t: str) -> bool:
        return t in ("int", "float")

    def is_numeric_or_unknown(self, t: str) -> bool:
        return t in ("int", "float", "unknown")

    def compatible_types(self, left: str, right: str) -> bool:
        if left == right:
            return True
        if left == "float" and right == "int":
            return True
        return False

    def visit_Program(self, node: ASTNode):
        for child in node.children:
            self.analyze(child)

    def visit_Block(self, node: ASTNode):
        self.enter_scope()
        for child in node.children:
            self.analyze(child)
        self.exit_scope()

    def visit_VarDecl(self, node: ASTNode):
        decl_kind = node.value
        var_name = node.children[0].value
        if len(node.children) == 1:
            if decl_kind == "const":
                raise SemanticError(f"Semantic error: const variable '{var_name}' must be initialized")
            self.current_scope.declare(Symbol(var_name, "unknown", decl_kind, initialized=False))
            return

        expr_type = self.analyze(node.children[1])
        self.current_scope.declare(Symbol(var_name, expr_type, decl_kind, initialized=True))

    def visit_Identifier(self, node: ASTNode):
        symbol = self.current_scope.lookup(node.value)
        if symbol is None:
            raise SemanticError(f"Semantic error: variable '{node.value}' used before declaration")
        return symbol.var_type

    def visit_Literal(self, node: ASTNode):
        return self.get_literal_type(node.value)

    def visit_Assign(self, node: ASTNode):
        op = node.value
        left = node.children[0]
        right = node.children[1]
        if left.kind != "Identifier":
            raise SemanticError("Semantic error: invalid assignment target")
        symbol = self.current_scope.lookup(left.value)
        if symbol is None:
            raise SemanticError(f"Semantic error: variable '{left.value}' assigned before declaration")
        if symbol.kind == "const":
            raise SemanticError(f"Semantic error: const variable '{left.value}' cannot be reassigned")

        right_type = self.analyze(right)
        if op == "=":
            if symbol.var_type == "unknown":
                symbol.var_type = right_type
            elif not self.compatible_types(symbol.var_type, right_type):
                raise SemanticError(
                    f"Semantic error: cannot assign {right_type} to '{left.value}' of type {symbol.var_type}"
                )
            symbol.initialized = True
            return symbol.var_type

        if op in ("+=", "-=", "*=", "/="):
            if not self.is_numeric_or_unknown(symbol.var_type) or not self.is_numeric_or_unknown(right_type):
                raise SemanticError(f"Semantic error: operator '{op}' requires numeric operands")
            if "unknown" in (symbol.var_type, right_type):
                return "unknown"
            return "float" if "float" in (symbol.var_type, right_type) else "int"

    def visit_BinaryOp(self, node: ASTNode):
        op = node.value
        left_type = self.analyze(node.children[0])
        right_type = self.analyze(node.children[1])

        if op in ("+", "-", "*", "/", "%"):
            if not self.is_numeric_or_unknown(left_type) or not self.is_numeric_or_unknown(right_type):
                raise SemanticError(f"Semantic error: arithmetic operator '{op}' requires numeric operands")
            if "unknown" in (left_type, right_type):
                return "unknown"
            if op == "/":
                return "float"
            return "float" if "float" in (left_type, right_type) else "int"

        if op in ("<", ">", "<=", ">="):
            if not self.is_numeric_or_unknown(left_type) or not self.is_numeric_or_unknown(right_type):
                raise SemanticError(f"Semantic error: comparison operator '{op}' requires numeric operands")
            return "bool"

        if op in ("==", "!=", "==="):
            if not self.compatible_types(left_type, right_type) and not self.compatible_types(right_type, left_type):
                raise SemanticError(f"Semantic error: cannot compare {left_type} with {right_type}")
            return "bool"

        if op in ("&&", "||"):
            if left_type != "bool" or right_type != "bool":
                raise SemanticError(f"Semantic error: logical operator '{op}' requires boolean operands")
            return "bool"

        return "unknown"

    def visit_UnaryOp(self, node: ASTNode):
        op = node.value
        expr = node.children[0]
        expr_type = self.analyze(expr)

        if op == "!":
            if expr_type != "bool":
                raise SemanticError("Semantic error: '!' operator requires boolean operand")
            return "bool"

        if op == "-":
            if not self.is_numeric(expr_type):
                raise SemanticError("Semantic error: unary '-' requires numeric operand")
            return expr_type

        if op in ("++", "--"):
            if expr.kind != "Identifier":
                raise SemanticError(f"Semantic error: '{op}' requires a variable")
            symbol = self.current_scope.lookup(expr.value)
            if symbol is None:
                raise SemanticError(f"Semantic error: variable '{expr.value}' used before declaration")
            if symbol.kind == "const":
                raise SemanticError(f"Semantic error: const variable '{expr.value}' cannot be updated")
            if not self.is_numeric_or_unknown(expr_type):
                raise SemanticError(f"Semantic error: '{op}' requires numeric operand")
            return expr_type

    def visit_PostfixOp(self, node: ASTNode):
        expr = node.children[0]
        expr_type = self.analyze(expr)
        if expr.kind != "Identifier":
            raise SemanticError(f"Semantic error: postfix '{node.value}' requires a variable")
        symbol = self.current_scope.lookup(expr.value)
        if symbol is None:
            raise SemanticError(f"Semantic error: variable '{expr.value}' used before declaration")
        if symbol.kind == "const":
            raise SemanticError(f"Semantic error: const variable '{expr.value}' cannot be updated")
        if not self.is_numeric_or_unknown(expr_type):
            raise SemanticError(f"Semantic error: postfix '{node.value}' requires numeric operand")
        return expr_type

    def visit_Print(self, node: ASTNode):
        self.analyze(node.children[0])

    def visit_ExpressionStatement(self, node: ASTNode):
        self.analyze(node.children[0])

    def visit_If(self, node: ASTNode):
        cond_type = self.analyze(node.children[0])
        if cond_type != "bool":
            raise SemanticError("Semantic error: if condition must be boolean")
        self.analyze(node.children[1])
        if len(node.children) > 2:
            self.analyze(node.children[2])

    def visit_While(self, node: ASTNode):
        cond_type = self.analyze(node.children[0])
        if cond_type != "bool":
            raise SemanticError("Semantic error: while condition must be boolean")
        self.analyze(node.children[1])

    def visit_For(self, node: ASTNode):
        self.enter_scope()
        init, cond, update, body = node.children
        if init.kind != "EmptyInit":
            self.analyze(init)
        if cond.kind != "EmptyCondition":
            cond_type = self.analyze(cond)
            if cond_type != "bool":
                raise SemanticError("Semantic error: for condition must be boolean")
        if update.kind != "EmptyUpdate":
            self.analyze(update)
        self.analyze(body)
        self.exit_scope()

    def visit_FunctionDecl(self, node: ASTNode):
        func_name = node.value
        params_node, body_node = node.children
        param_names = [p.value for p in params_node.children]
        self.current_scope.declare(Symbol(func_name, "function", "function", initialized=True, params=param_names))

        self.enter_scope()
        old_in_function = self.in_function
        self.in_function = True
        for param in param_names:
            self.current_scope.declare(Symbol(param, "unknown", "param", initialized=True))
        self.analyze(body_node)
        self.in_function = old_in_function
        self.exit_scope()

    def visit_Return(self, node: ASTNode):
        if not self.in_function:
            raise SemanticError("Semantic error: return statement outside function")
        if node.children:
            return self.analyze(node.children[0])
        return "null"

    def visit_Call(self, node: ASTNode):
        callee = node.children[0]
        args_node = node.children[1]
        symbol = self.current_scope.lookup(callee.value)
        if symbol is None or symbol.kind != "function":
            raise SemanticError(f"Semantic error: '{callee.value}' is not a declared function")
        if symbol.params is not None and len(args_node.children) != len(symbol.params):
            raise SemanticError(
                f"Semantic error: function '{callee.value}' expects {len(symbol.params)} argument(s), "
                f"got {len(args_node.children)}"
            )
        for arg in args_node.children:
            self.analyze(arg)
        return "unknown"


# =========================
# Code Generation
# =========================
class CodeGenerator:
    def __init__(self):
        self.tac: List[str] = []
        self.pseudo: List[str] = []
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self) -> str:
        self.temp_count += 1
        return f"t{self.temp_count}"

    def new_label(self, prefix: str = "L") -> str:
        self.label_count += 1
        return f"{prefix}{self.label_count}"

    def emit_tac(self, line: str):
        self.tac.append(line)

    def emit_pseudo(self, line: str):
        self.pseudo.append(line)

    def generate(self, node: ASTNode) -> Tuple[List[str], List[str]]:
        self.visit(node)
        return self.tac, self.pseudo

    def visit(self, node: ASTNode):
        method = getattr(self, f"visit_{node.kind}", self.generic_visit)
        return method(node)

    def generic_visit(self, node: ASTNode):
        for child in node.children:
            self.visit(child)

    def visit_Program(self, node: ASTNode):
        for child in node.children:
            self.visit(child)

    def visit_Block(self, node: ASTNode):
        for child in node.children:
            self.visit(child)

    def visit_VarDecl(self, node: ASTNode):
        name = node.children[0].value
        if len(node.children) > 1:
            value = self.gen_expr(node.children[1])
            self.emit_tac(f"{name} = {value}")
            self.emit_pseudo(f"STORE {name}, {value}")
        else:
            self.emit_tac(f"declare {name}")
            self.emit_pseudo(f"DECLARE {name}")

    def visit_Print(self, node: ASTNode):
        value = self.gen_expr(node.children[0])
        self.emit_tac(f"print {value}")
        self.emit_pseudo(f"PRINT {value}")

    def visit_ExpressionStatement(self, node: ASTNode):
        self.gen_expr(node.children[0])

    def visit_If(self, node: ASTNode):
        cond = self.gen_expr(node.children[0])
        else_label = self.new_label("ELSE")
        end_label = self.new_label("ENDIF")
        self.emit_tac(f"if_false {cond} goto {else_label}")
        self.emit_pseudo(f"JZ {cond}, {else_label}")
        self.visit(node.children[1])
        self.emit_tac(f"goto {end_label}")
        self.emit_pseudo(f"JMP {end_label}")
        self.emit_tac(f"label {else_label}")
        self.emit_pseudo(f"LABEL {else_label}")
        if len(node.children) > 2:
            self.visit(node.children[2])
        self.emit_tac(f"label {end_label}")
        self.emit_pseudo(f"LABEL {end_label}")

    def visit_While(self, node: ASTNode):
        start = self.new_label("WHILE")
        end = self.new_label("ENDWHILE")
        self.emit_tac(f"label {start}")
        self.emit_pseudo(f"LABEL {start}")
        cond = self.gen_expr(node.children[0])
        self.emit_tac(f"if_false {cond} goto {end}")
        self.emit_pseudo(f"JZ {cond}, {end}")
        self.visit(node.children[1])
        self.emit_tac(f"goto {start}")
        self.emit_pseudo(f"JMP {start}")
        self.emit_tac(f"label {end}")
        self.emit_pseudo(f"LABEL {end}")

    def visit_For(self, node: ASTNode):
        init, cond, update, body = node.children
        start = self.new_label("FOR")
        end = self.new_label("ENDFOR")
        if init.kind != "EmptyInit":
            self.visit(init)
        self.emit_tac(f"label {start}")
        self.emit_pseudo(f"LABEL {start}")
        if cond.kind != "EmptyCondition":
            c = self.gen_expr(cond)
            self.emit_tac(f"if_false {c} goto {end}")
            self.emit_pseudo(f"JZ {c}, {end}")
        self.visit(body)
        if update.kind != "EmptyUpdate":
            self.gen_expr(update)
        self.emit_tac(f"goto {start}")
        self.emit_pseudo(f"JMP {start}")
        self.emit_tac(f"label {end}")
        self.emit_pseudo(f"LABEL {end}")

    def visit_FunctionDecl(self, node: ASTNode):
        name = node.value
        params = [p.value for p in node.children[0].children]
        self.emit_tac(f"func {name}({', '.join(params)})")
        self.emit_pseudo(f"FUNC {name} {', '.join(params)}")
        self.visit(node.children[1])
        self.emit_tac(f"endfunc {name}")
        self.emit_pseudo(f"END_FUNC {name}")

    def visit_Return(self, node: ASTNode):
        if node.children:
            value = self.gen_expr(node.children[0])
            self.emit_tac(f"return {value}")
            self.emit_pseudo(f"RET {value}")
        else:
            self.emit_tac("return")
            self.emit_pseudo("RET")

    def gen_expr(self, node: ASTNode) -> str:
        method = getattr(self, f"expr_{node.kind}", None)
        if method is None:
            raise ValueError(f"Code generation not implemented for {node.kind}")
        return method(node)

    def expr_Literal(self, node: ASTNode) -> str:
        value = node.value
        if value in ("true", "false", "null"):
            return str(value)
        if isinstance(value, str):
            try:
                if "." not in value and "e" not in value.lower():
                    int(value)
                    return value
            except Exception:
                pass
            try:
                if "." in value or "e" in value.lower():
                    float(value)
                    return value
            except Exception:
                pass
            return repr(value)
        return str(value)

    def expr_Identifier(self, node: ASTNode) -> str:
        return node.value

    def expr_Assign(self, node: ASTNode) -> str:
        left = node.children[0].value
        right = self.gen_expr(node.children[1])
        op = node.value
        if op == "=":
            self.emit_tac(f"{left} = {right}")
            self.emit_pseudo(f"STORE {left}, {right}")
        else:
            base_op = op[0]
            temp = self.new_temp()
            self.emit_tac(f"{temp} = {left} {base_op} {right}")
            self.emit_tac(f"{left} = {temp}")
            self.emit_pseudo(f"LOAD {left}")
            self.emit_pseudo(f"LOAD {right}")
            self.emit_pseudo(f"{self.op_to_instr(base_op)}")
            self.emit_pseudo(f"STORE {left}")
        return left

    def expr_BinaryOp(self, node: ASTNode) -> str:
        left = self.gen_expr(node.children[0])
        right = self.gen_expr(node.children[1])
        temp = self.new_temp()
        self.emit_tac(f"{temp} = {left} {node.value} {right}")
        op_name = self.op_to_instr(node.value)
        self.emit_pseudo(f"LOAD {left}")
        self.emit_pseudo(f"LOAD {right}")
        self.emit_pseudo(f"{op_name}")
        self.emit_pseudo(f"STORE {temp}")
        return temp

    def expr_UnaryOp(self, node: ASTNode) -> str:
        expr = self.gen_expr(node.children[0])
        op = node.value
        if op in ("++", "--") and node.children[0].kind == "Identifier":
            name = node.children[0].value
            sign = "+" if op == "++" else "-"
            temp = self.new_temp()
            self.emit_tac(f"{temp} = {name} {sign} 1")
            self.emit_tac(f"{name} = {temp}")
            self.emit_pseudo(f"LOAD {name}")
            self.emit_pseudo("PUSH 1")
            self.emit_pseudo("ADD" if sign == "+" else "SUB")
            self.emit_pseudo(f"STORE {name}")
            return name
        temp = self.new_temp()
        self.emit_tac(f"{temp} = {op}{expr}")
        self.emit_pseudo(f"UNARY {op} {expr}")
        return temp

    def expr_PostfixOp(self, node: ASTNode) -> str:
        name = self.gen_expr(node.children[0])
        sign = "+" if node.value == "++" else "-"
        temp = self.new_temp()
        self.emit_tac(f"{temp} = {name}")
        self.emit_tac(f"{name} = {name} {sign} 1")
        self.emit_pseudo(f"STORE {name}, {name} {sign} 1")
        return temp

    def expr_Call(self, node: ASTNode) -> str:
        func_name = node.children[0].value
        args_node = node.children[1]
        for arg in args_node.children:
            value = self.gen_expr(arg)
            self.emit_tac(f"param {value}")
            self.emit_pseudo(f"PUSH {value}")
        temp = self.new_temp()
        self.emit_tac(f"{temp} = call {func_name}, {len(args_node.children)}")
        self.emit_pseudo(f"CALL {func_name}, {len(args_node.children)} -> {temp}")
        return temp

    @staticmethod
    def op_to_instr(op: str) -> str:
        mapping = {
            "+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "%": "MOD",
            "<": "LT", ">": "GT", "<=": "LE", ">=": "GE",
            "==": "EQ", "!=": "NE", "===": "SEQ",
            "&&": "AND", "||": "OR",
        }
        return mapping.get(op, f"OP_{op}")


# =========================
# Runtime / Demo Executor
# =========================
class RuntimeErrorMiniLang(Exception):
    pass


class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


@dataclass
class Binding:
    value: Any
    mutable: bool


class Environment:
    def __init__(self, parent: Optional["Environment"] = None):
        self.parent = parent
        self.values: Dict[str, Binding] = {}

    def define(self, name: str, value: Any, mutable: bool = True):
        if name in self.values:
            raise RuntimeErrorMiniLang(f"Runtime error: '{name}' already defined in this scope")
        self.values[name] = Binding(value, mutable)

    def resolve(self, name: str) -> Binding:
        env = self
        while env is not None:
            if name in env.values:
                return env.values[name]
            env = env.parent
        raise RuntimeErrorMiniLang(f"Runtime error: undefined variable '{name}'")

    def assign(self, name: str, value: Any):
        env = self
        while env is not None:
            if name in env.values:
                binding = env.values[name]
                if not binding.mutable:
                    raise RuntimeErrorMiniLang(f"Runtime error: const variable '{name}' cannot be reassigned")
                binding.value = value
                return
            env = env.parent
        raise RuntimeErrorMiniLang(f"Runtime error: undefined variable '{name}'")


@dataclass
class UserFunction:
    name: str
    params: List[str]
    body: ASTNode
    closure: Environment


class MiniRuntime:
    def __init__(self):
        self.global_env = Environment()
        self.output: List[str] = []

    def run(self, node: ASTNode) -> List[str]:
        self.exec_program(node, self.global_env)
        return self.output

    def exec_program(self, node: ASTNode, env: Environment):
        for child in node.children:
            self.exec_stmt(child, env)

    def exec_block(self, node: ASTNode, env: Environment):
        block_env = Environment(env)
        for child in node.children:
            self.exec_stmt(child, block_env)

    def exec_stmt(self, node: ASTNode, env: Environment):
        kind = node.kind

        if kind == "VarDecl":
            name = node.children[0].value
            value = self.eval_expr(node.children[1], env) if len(node.children) > 1 else None
            env.define(name, value, mutable=(node.value != "const"))
            return

        if kind == "Print":
            value = self.eval_expr(node.children[0], env)
            self.output.append(str(value))
            return

        if kind == "ExpressionStatement":
            self.eval_expr(node.children[0], env)
            return

        if kind == "Block":
            self.exec_block(node, env)
            return

        if kind == "If":
            cond = self.eval_expr(node.children[0], env)
            if cond:
                self.exec_stmt(node.children[1], env)
            elif len(node.children) > 2:
                self.exec_stmt(node.children[2], env)
            return

        if kind == "While":
            while self.eval_expr(node.children[0], env):
                self.exec_stmt(node.children[1], env)
            return

        if kind == "For":
            loop_env = Environment(env)
            init, cond, update, body = node.children
            if init.kind != "EmptyInit":
                self.exec_stmt(init, loop_env) if init.kind == "VarDecl" else self.eval_expr(init, loop_env)
            while True:
                if cond.kind != "EmptyCondition" and not self.eval_expr(cond, loop_env):
                    break
                self.exec_stmt(body, loop_env)
                if update.kind != "EmptyUpdate":
                    self.eval_expr(update, loop_env)
            return

        if kind == "FunctionDecl":
            params = [p.value for p in node.children[0].children]
            env.define(node.value, UserFunction(node.value, params, node.children[1], env), mutable=False)
            return

        if kind == "Return":
            value = self.eval_expr(node.children[0], env) if node.children else None
            raise ReturnSignal(value)

        if kind == "Program":
            self.exec_program(node, env)
            return

        raise RuntimeErrorMiniLang(f"Runtime error: unsupported statement '{kind}'")

    def eval_expr(self, node: ASTNode, env: Environment):
        kind = node.kind

        if kind == "Literal":
            return self.literal_value(node.value)

        if kind == "Identifier":
            return env.resolve(node.value).value

        if kind == "Assign":
            name = node.children[0].value
            op = node.value
            right = self.eval_expr(node.children[1], env)
            if op == "=":
                env.assign(name, right)
                return right
            current = env.resolve(name).value
            result = self.apply_binary(op[0], current, right)
            env.assign(name, result)
            return result

        if kind == "BinaryOp":
            left = self.eval_expr(node.children[0], env)
            right = self.eval_expr(node.children[1], env)
            return self.apply_binary(node.value, left, right)

        if kind == "UnaryOp":
            op = node.value
            child = node.children[0]
            value = self.eval_expr(child, env)
            if op == "!":
                return not value
            if op == "-":
                return -value
            if op in ("++", "--"):
                if child.kind != "Identifier":
                    raise RuntimeErrorMiniLang(f"Runtime error: '{op}' requires identifier")
                name = child.value
                new_value = value + 1 if op == "++" else value - 1
                env.assign(name, new_value)
                return new_value

        if kind == "PostfixOp":
            child = node.children[0]
            if child.kind != "Identifier":
                raise RuntimeErrorMiniLang(f"Runtime error: postfix '{node.value}' requires identifier")
            name = child.value
            old_value = env.resolve(name).value
            env.assign(name, old_value + 1 if node.value == "++" else old_value - 1)
            return old_value

        if kind == "Call":
            func_name = node.children[0].value
            func_obj = env.resolve(func_name).value
            if not isinstance(func_obj, UserFunction):
                raise RuntimeErrorMiniLang(f"Runtime error: '{func_name}' is not callable")
            args = [self.eval_expr(arg, env) for arg in node.children[1].children]
            return self.call_function(func_obj, args)

        raise RuntimeErrorMiniLang(f"Runtime error: unsupported expression '{kind}'")

    def call_function(self, func: UserFunction, args: List[Any]):
        if len(args) != len(func.params):
            raise RuntimeErrorMiniLang(
                f"Runtime error: function '{func.name}' expects {len(func.params)} argument(s), got {len(args)}"
            )
        call_env = Environment(func.closure)
        for name, value in zip(func.params, args):
            call_env.define(name, value, mutable=True)
        try:
            self.exec_stmt(func.body, call_env)
        except ReturnSignal as signal:
            return signal.value
        return None

    @staticmethod
    def literal_value(value: Any):
        if value == "true":
            return True
        if value == "false":
            return False
        if value == "null":
            return None
        if isinstance(value, str):
            try:
                if "." not in value and "e" not in value.lower():
                    return int(value)
            except Exception:
                pass
            try:
                if "." in value or "e" in value.lower():
                    return float(value)
            except Exception:
                pass
            return value
        return value

    @staticmethod
    def apply_binary(op: str, left: Any, right: Any):
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "%":
            return left % right
        if op == "<":
            return left < right
        if op == ">":
            return left > right
        if op == "<=":
            return left <= right
        if op == ">=":
            return left >= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "===":
            return type(left) == type(right) and left == right
        if op == "&&":
            return bool(left) and bool(right)
        if op == "||":
            return bool(left) or bool(right)
        raise RuntimeErrorMiniLang(f"Runtime error: unknown operator '{op}'")


# =========================
# Compiler Facade
# =========================
SAMPLE_PROGRAM = r'''
fn add(a, b) {
    return a + b;
}

let x = 10;
let y = 20;
let total = add(x, y);
print(total);

if (total > 20) {
    print("greater than twenty");
} else {
    print("small value");
}

for (let i = 0; i < 3; i++) {
    print(i);
}
'''


def compile_source(source: str) -> Dict[str, Any]:
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    parser = Parser(tokens)
    ast = parser.parse()

    analyzer = SemanticAnalyzer()
    analyzer.analyze(ast)

    generator = CodeGenerator()
    tac, pseudo = generator.generate(ast)

    runtime = MiniRuntime()
    output = runtime.run(ast)

    return {
        "tokens": [repr(t) for t in tokens],
        "lexical_table": lexer.lexical_table(),
        "ast": ast.pretty(),
        "three_address_code": tac,
        "pseudo_code": pseudo,
        "execution_output": output,
    }


def format_report(result: Dict[str, Any]) -> str:
    parts = []
    parts.append("MINILANG COMPILER - FINAL DEMO\n")
    parts.append("=" * 60)
    parts.append("\n1. TOKENS\n")
    parts.extend(result["tokens"])
    parts.append("\n\n2. LEXICAL TABLE\n")
    parts.append(json.dumps(result["lexical_table"], indent=2))
    parts.append("\n\n3. AST\n")
    parts.append(result["ast"])
    parts.append("\n4. SEMANTIC ANALYSIS\n")
    parts.append("Semantic analysis successful")
    parts.append("\n\n5. THREE-ADDRESS CODE\n")
    parts.extend(result["three_address_code"])
    parts.append("\n\n6. PSEUDO CODE\n")
    parts.extend(result["pseudo_code"])
    parts.append("\n\n7. EXECUTION OUTPUT\n")
    parts.extend(result["execution_output"])
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="MiniLang Compiler Final Product")
    parser.add_argument("source", nargs="?", help="MiniLang source file")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo program")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    if args.demo or not args.source:
        source = SAMPLE_PROGRAM
    else:
        with open(args.source, "r", encoding="utf-8") as f:
            source = f.read()

    try:
        result = compile_source(source)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_report(result))
    except (LexerError, ParserError, SemanticError, RuntimeErrorMiniLang) as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
