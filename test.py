import abc

# class MediaLoader(abc.ABC):
#     @abc.abstractmethod
#     def play(self) -> None:
#         ...

# class Wav(MediaLoader):
#     pass


# class Ogg(MediaLoader): 
#     def play(self): 
#         print('hello world')
# o = Ogg()
# o.play()
# x = Wav()

class TestABC(abc.ABC):
    @abc.abstractmethod
    def bry(self, a, b):
        pass

class T1ABC(TestABC):
    def bry(self, a, b):
        print(f'hello world: {a}, {b}')

class T2ABC(TestABC):
    def bry(self, a):
        print(f'hello world {a}')

class T3ABC(TestABC):
    def bry(self):
        pass

class T4ABC(TestABC):
    def abry(self):
        pass

t1 = T1ABC()
t1.bry(1,2)

t2 = T2ABC()
t2.bry(2)

t3 = T3ABC()
t3.bry()

t4 = T4ABC()