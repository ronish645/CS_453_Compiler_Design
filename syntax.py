from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any

# Import from your lexer file
# Make sure final.py contains Token, Lexer, LexerError
from final import Token, Lexer, LexerError


# =========================
# Parser Error
# =========================
class ParserError(Exception):
    pass


# =========================
# AST Node
# =========================
@dataclass
class ASTNode:
    kind: str
    value: Any = None
    children: List["ASTNode"] = field(default_factory=list)

    def pretty(self, level=0) -> str:
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

    # ---------- helpers ----------
    def current(self) -> Token:
        return self.tokens[self.i]

    def peek(self, k: int = 1) -> Token:
        j = self.i + k
        if j < len(self.tokens):
            return self.tokens[j]
        return self.tokens[-1]

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

    # ---------- entry ----------
    def parse(self) -> ASTNode:
        program = ASTNode("Program")
        while self.current().type != "EOF":
            program.children.append(self.statement())
        return program

    # ---------- statements ----------
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

        # assignment or expression statement
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
        kind = self.expect("KW").value  # let or const
        ident = self.expect("IDENT").value

        init = None
        if self.match("OP", "="):
            init = self.expression()

        self.expect("SEP", ";")
        node = ASTNode("VarDecl", value=kind)
        node.children.append(ASTNode("Identifier", value=ident))
        if init:
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
        node.children.append(condition)
        node.children.append(then_branch)

        if self.match("KW", "else"):
            else_branch = self.statement()
            node.children.append(else_branch)

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
        if init:
            node.children.append(init)
        else:
            node.children.append(ASTNode("EmptyInit"))
        if condition:
            node.children.append(condition)
        else:
            node.children.append(ASTNode("EmptyCondition"))
        if update:
            node.children.append(update)
        else:
            node.children.append(ASTNode("EmptyUpdate"))
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
        if init:
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
        node.children.append(params)
        node.children.append(body)
        return node

    def return_statement(self) -> ASTNode:
        self.expect("KW", "return")
        if self.current().type == "SEP" and self.current().value == ";":
            self.expect("SEP", ";")
            return ASTNode("Return")
        expr = self.expression()
        self.expect("SEP", ";")
        return ASTNode("Return", children=[expr])

    # ---------- expressions ----------
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

            # function call
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
# Example test
# =========================
if __name__ == "__main__":
    code = r"""
    fn add(a, b) {
        return a + b;
    }

    let x = 10;
    let y = 20;
    print(add(x, y));

    if (x < y) {
        print("x is smaller");
    } else {
        print("y is smaller");
    }

    for (let i = 0; i < 5; i++) {
        print(i);
    }
    """

    try:
        lexer = Lexer(code)
        tokens = lexer.tokenize()

        parser = Parser(tokens)
        ast = parser.parse()

        print("SYNTAX ANALYSIS SUCCESSFUL\n")
        print(ast.pretty())

    except (LexerError, ParserError) as e:
        print("ERROR:", e)