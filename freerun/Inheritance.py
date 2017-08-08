class Animal(object):
    def run(self):
        print('Animal is running')

class Dog(Animal):
    pass

class Cat(Animal):
    pass

dog = Dog()
cat = Cat()

dog.run()
cat.run()

class Dog(Animal):
    def run(self):
        print('Dog is running')

    def eat(self):
        print('Eating meat')

class Cat(Animal):
    def run(self):
        print('Cat is running')

    def eat(self):
        print('Eating meat')

def run_twice(animal):
    animal.run()
    animal.run()

# with inheritance, we can give run_twice any class
# as long as that class is the subclass of the father class
# if we do not make use of the inheritance, every time we
# need to define a new run_twice