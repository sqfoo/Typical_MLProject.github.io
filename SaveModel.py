
fn = open("OriginalHousingBoston.txt", 'r')
fn1 = open("Modified.txt", 'w')
cont = fn.readlines()

for i in range(22):
    fn1.write(cont[i])

length = int((len(cont)-22)/2)
a = cont[1].replace('\n', '')  + cont[2]
print((a))
for i in range(length):

    fn1.write(cont[22+2*i].replace('\n', '')+cont[22+2*i+1])
fn.close()
fn1.close()