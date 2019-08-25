rlt = []
with open('p_mulrel.txt', 'r') as fi:
    for i, line in enumerate(fi.readlines()):
        if (i % 5 == 0):
            rlt.append(int(line.strip()))

with open('p_mulrel_filter.txt', 'w') as fo:
    for i, r in enumerate(rlt):
        if i>=2 and i<=len(rlt)-1-2:
            p = (rlt[i-2]+rlt[i-1]+rlt[i]+rlt[i+1]+rlt[i+2])//5
            fo.write(str(p)+'\n')
        else:
            fo.write(str(rlt[i]) + '\n')




rlt = []
with open('p_dca.txt', 'r') as fi:
    for i, line in enumerate(fi.readlines()):
        if (i % 5 == 0):
            rlt.append(int(line.strip()))

with open('p_dca_filter.txt', 'w') as fo:
    for i, r in enumerate(rlt):
        if i>=2 and i<=len(rlt)-1-2:
            p = (rlt[i-2]+rlt[i-1]+rlt[i]+rlt[i+1]+rlt[i+2])//5
            fo.write(str(p)+'\n')
        else:
            fo.write(str(rlt[i]) + '\n')
