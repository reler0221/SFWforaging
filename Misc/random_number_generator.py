import random as rand

pool = list(range(446))

draws = 20

output = rand.sample(pool, 20)
print(sorted(output))



