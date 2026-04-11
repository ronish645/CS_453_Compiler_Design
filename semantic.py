from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# your files
from final import Lexer, LexerError
from syntax import Parser, ParserError, ASTNode


# =========================
# Symbol + Scope
# =========================
@dataclass
class Symbol:
    name: str
    sym_type: str
    is_const: bool = False
    initialized: bool = False
    is_function: bool = False
    params: Optional[List[str]] = None


class Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}

    def declare(self, symbol: Symbol):
        if symbol.name in self.symbols:
            raise Exception(f"Redeclaration of '{symbol.name}'")
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        scope = self
        while scope:
            if name in scope.symbols:
                return scope.symbols[name]
            scope = scope.parent
        return None


# =========================
# Semantic Analyzer
# =========================
class SemanticAnalyzer:
    def __init__(self):
        self.global_scope = Scope()
        self.current_scope = self.global_scope
        self.errors: List[str] = []
        self.current_function = None

    def error(self, msg):
        self.errors.append(msg)

    def enter_scope(self):
        self.current_scope = Scope(self.current_scope)

    def exit_scope(self):
        self.current_scope = self.current_scope.parent

    def analyze(self, node: ASTNode):
        self.visit(node)
        return self.errors

    def visit(self, node):
        if node is None:
            return None
        method = getattr(self, f"visit_{node.kind}", self.generic)
        return method(node)

    def generic(self, node):
        for c in node.children:
            self.visit(c)

    # ---------- type helpers ----------
    def infer_type(self, value):
        if value in ("true", "false"):
            return "bool"
        if value == "null":
            return "null"
        try:
            int(value)
            return "int"
        except:
            pass
        try:
            float(value)
            return "float"
        except:
            pass
        return "string"

    def is_numeric(self, t):
        return t in ("int", "float")

    # ---------- program ----------
    def visit_Program(self, node):
        for c in node.children:
            self.visit(c)

    def visit_Block(self, node):
        self.enter_scope()
        for c in node.children:
            self.visit(c)
        self.exit_scope()

    # ---------- declarations ----------
    def visit_VarDecl(self, node):
        kind = node.value
        name = node.children[0].value
        is_const = (kind == "const")

        if len(node.children) > 1:
            t = self.visit(node.children[1])
        else:
            t = "unknown"

        try:
            self.current_scope.declare(Symbol(name, t, is_const, True))
        except Exception as e:
            self.error(str(e))

    def visit_FunctionDecl(self, node):
        name = node.value
        params = [p.value for p in node.children[0].children]

        try:
            self.current_scope.declare(Symbol(name, "function", True, True, True, params))
        except Exception as e:
            self.error(str(e))
            return

        self.enter_scope()
        for p in params:
            self.current_scope.declare(Symbol(p, "unknown", False, True))

        old = self.current_function
        self.current_function = name

        self.visit(node.children[1])

        self.current_function = old
        self.exit_scope()

    # ---------- statements ----------
    def visit_Print(self, node):
        self.visit(node.children[0])

    def visit_ExpressionStatement(self, node):
        self.visit(node.children[0])

    def visit_If(self, node):
        cond = self.visit(node.children[0])
        if cond != "bool":
            self.error("If condition must be boolean")
        self.visit(node.children[1])
        if len(node.children) > 2:
            self.visit(node.children[2])

    def visit_While(self, node):
        cond = self.visit(node.children[0])
        if cond != "bool":
            self.error("While condition must be boolean")
        self.visit(node.children[1])

    def visit_For(self, node):
        self.enter_scope()

        if node.children[0].kind != "EmptyInit":
            self.visit(node.children[0])

        if node.children[1].kind != "EmptyCondition":
            if self.visit(node.children[1]) != "bool":
                self.error("For condition must be boolean")

        if node.children[2].kind != "EmptyUpdate":
            self.visit(node.children[2])

        self.visit(node.children[3])
        self.exit_scope()

    def visit_Return(self, node):
        if self.current_function is None:
            self.error("Return outside function")

    # ---------- expressions ----------
    def visit_Identifier(self, node):
        sym = self.current_scope.lookup(node.value)
        if not sym:
            self.error(f"Variable '{node.value}' not declared")
            return "unknown"
        return sym.sym_type

    def visit_Literal(self, node):
        return self.infer_type(node.value)

    def visit_Assign(self, node):
        name = node.children[0].value
        sym = self.current_scope.lookup(name)

        if not sym:
            self.error(f"Variable '{name}' not declared")
            return "unknown"

        if sym.is_const:
            self.error(f"Cannot reassign const '{name}'")

        t = self.visit(node.children[1])
        sym.sym_type = t
        return t

    def visit_BinaryOp(self, node):
        l = self.visit(node.children[0])
        r = self.visit(node.children[1])
        op = node.value

        if op in ("+", "-", "*", "/", "%"):
            if self.is_numeric(l) and self.is_numeric(r):
                return "float" if "float" in (l, r) else "int"
            self.error("Arithmetic needs numbers")
            return "unknown"

        if op in ("<", ">", "<=", ">="):
            return "bool"

        if op in ("&&", "||"):
            if l == "bool" and r == "bool":
                return "bool"
            self.error("Logical needs bool")
            return "bool"

        return "unknown"

    def visit_UnaryOp(self, node):
        return self.visit(node.children[0])

    def visit_PostfixOp(self, node):
        return self.visit(node.children[0])

    def visit_Call(self, node):
        name = node.children[0].value
        sym = self.current_scope.lookup(name)

        if not sym or not sym.is_function:
            self.error(f"{name} is not a function")
            return "unknown"

        args = node.children[1].children if len(node.children) > 1 else []
        if len(args) != len(sym.params):
            self.error(f"{name} argument count mismatch")

        for a in args:
            self.visit(a)

        return "unknown"


# =========================
# TESTING PART
# =========================
if __name__ == "__main__":

    code = r"""
    fn add(a, b) {
        return a + b;
    }

    let x = 10;
    const y = 20;

    print(add(x, y));

    if (x < y) {
        print("ok");
    }

    x = x + 1;
    y = 30;   // ERROR
    z = 5;    // ERROR
    """

    try:
        lexer = Lexer(code)
        tokens = lexer.tokenize()

        parser = Parser(tokens)
        ast = parser.parse()

        print("=== AST ===")
        print(ast.pretty())

        analyzer = SemanticAnalyzer()
        errors = analyzer.analyze(ast)

        print("\n=== SEMANTIC RESULT ===")
        if errors:
            for e in errors:
                print("-", e)
        else:
            print("No semantic errors!")

    except (LexerError, ParserError) as e:
        print("ERROR:", e)