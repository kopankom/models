import random
def getRandomNumbers():
    return random.randint(1, 15000), random.randint(15000, 40000)

def getNumberList(n=1000):
    result = ''
    for i in xrange(n):
        val1, val2 = getRandomNumbers()
        result += "{};{};{}\n".format(val1, val2, val1+val2)

    return result

def getConsecutiveNumbers(n=100):
    result = ''
    for i in xrange(n):
        val1 = i
        val2 = i
        result += "{};{};{}\n".format(val1, val2, val1+val2)
        result += "{};{};{}\n".format(val1, 1, val1+1)

    return result

file = open('training_data.csv', 'w')
file.write("val1;val2;result\n")
file.write(getConsecutiveNumbers(200))
file.write(getNumberList(5000))
file.close()


file = open('test_data.csv', 'w')
file.write("val1;val2;result\n")
file.write(getConsecutiveNumbers(200))
file.write(getNumberList(5000))
file.close()
