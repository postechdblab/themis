
class Code:

    def __init__(self):
        self.lines = []

    def __str__(self):
        return '\n'.join(self.lines)

    def add(self, c):
        if isinstance(c, Code):
            if len(c.lines) > 0:
                self.lines = self.lines + c.lines
        elif isinstance(c, list):
            self.lines = self.lines + c
        elif isinstance(c, str):
            if len(c) > 0:
                self.lines.append(c)
        else:
            print(c)
            assert(False)        

    def merge(self, code):
        self.lines = code.lines + lines

    def addAll(self, lines):
        self.lines = self.lines + lines

    def addTabs(self, n):
        self.lines = list(map(lambda x: ('\t'* n) + x, self.lines))

    def toString(self):
        return '\n'.join(self.lines)

    def isEmpty(self):
        return len(self.lines) == 0