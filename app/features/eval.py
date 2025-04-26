def processAnswersFeatureV1(answers):
    # cdd
    return 0;

def processAnswersFeatureV2(answers):
    fht = answers.get('fht')
    kmsi = answers.get('kmsi')
    
    print("fht", fht)
    print("kmsi", kmsi)

    fhtScore = []
    for i in range(len(fht)):
        value = int(fht[i])
        if value == 1:
            fhtScore.append(3)
        elif value == 2:
            fhtScore.append(2)
        elif value == 3:
            fhtScore.append(1)

    kmsiScore = []
    for i in range(len(kmsi)):
        value = int(kmsi[i])
        if i <= 5:
            if value == 1:
                kmsiScore.append(1)
            elif value == 2:
                kmsiScore.append(3)
        if i > 5 and i <= 7: 
            if value == 1:
                kmsiScore.append(3)
            elif value == 2:
                kmsiScore.append(1)

    print("fhtScore", fhtScore)
    print("kmsiScore", kmsiScore)
    
    sumFht = sum(fhtScore)
    sumKmsi = sum(kmsiScore)
    totalScore = sumFht + sumKmsi
    
    if totalScore >= 25:
        return 0
    else:
        return 1
    
def processAnswersFeatureV3(answers):
    # comming soon
    return 0;