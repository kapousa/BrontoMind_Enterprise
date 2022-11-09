#  Copyright (c) 2021. Slonos Labs. All rights Reserved.

class Robot:
    counter = 0

    @staticmethod
    def sayHello(xx):
        return "Hi, I am " + str(xx)


Robot2 = type("Robot")

x = Robot()
print(x.counter)
print(Robot2.sayHello('A'))