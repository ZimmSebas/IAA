import math
import sys


def std_dev(d, c):
    return c * math.sqrt(d)


def lectura_datos():
    d = int(input("Ingrese valor entero d\n"))
    n = int(input("Ingrese valor entero n\n"))
    c = float(input("Ingrese valor real C:\n"))

    print("Ingrese d puntos d dimensionales, con sus valores separados por espacios")

    puntos = []

    for i in range(d):
        s = input()
        s = s.split()

        if len(s) != d:
            print("Error tamaño distinto de d")
            continue
        puntos.append(s)

    return (d, n, c, puntos)


def centros_ej1(d):
    c_1 = []
    c_2 = []

    for i in range(d):
        c_1.append(1)
        c_2.append(-1)

    return (c_1, c_2)


def centros_ej2(d):
    c_1 = []
    c_2 = []

    for i in range(d):
        if not (i):
            c_1.append(1)
            c_2.append(-1)
        else:
            c_1.append(0)
            c_2.append(0)

    return (c_1, c_2)


def elige_clase(p, c_1, c_2):
    # Deberia elegir a qué clase corresponde cada punto. Si supiera como
    return
