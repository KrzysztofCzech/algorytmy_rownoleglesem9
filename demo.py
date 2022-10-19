n = int(input('n = '))
if n < 2:
    print("Brak liczb pierwszych w podanym zakresie")
else:
    skr = [False] * (n + 1)
    i = 2
    while i * i <= n:
        if  not skr[i]:
            j = i * i            
            while j <= n:
                skr[j] = True
                j += i
        i += 1
    for i in range(2, n+1):
        if not skr[i]:
            print(i, end = " ")