def item_order(order):
    salad_count = 0
    hamburger_count = 0
    water_count = 0
    for i in range(len(order)):
        if order[i:i+len('salad')] == 'salad':
            salad_count += 1
        elif order[i:i+len('hamburger')] == 'hamburger':
            hamburger_count += 1
        elif order[i:i+len('water')] == 'water':
            water_count += 1
    return 'salad:' + str(salad_count) + ' hamburger:' + str(hamburger_count) + ' water:' + str(water_count)