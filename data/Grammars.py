import re


def tomita1(expr):
    return "0" not in expr


def tomita2(expr):
    return expr == "10" * (int(len(expr) / 2))


def tomita3(expr):
    # words containing an odd series of consecutive ones and then later an odd series of consecutive zeros
    pattern = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$")
    # tomita 3: opposite of the pattern
    return pattern.match(expr) is None


def tomita4(expr):
    return "000" not in expr


def tomita5(expr):
    return (expr.count("0") % 2 == 0) and (expr.count("1") % 2 == 0)


def tomita6(expr):
    return ((expr.count("0") - expr.count("1")) % 3) == 0


def tomita7(expr):
    return expr.count("10") <= 1


def rule1(expr):
    return "11111" in expr


def rule2(expr):
    pattern = re.compile("(1)+0(1)+01")
    return pattern.match(expr) is not None
