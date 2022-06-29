from abc import abstractmethod, ABC


class Task(ABC):
    """An abstract class that handles different tasks.
    Its abstract method is 'operate', which all tasks have to implement."""

    @abstractmethod
    def operate(self, *args):
        pass


class XorOperator(Task):
    def operate(self, x1, x2):
        return x1 ^ x2


class AddOperator(Task):
    def operate(self, x1, x2):
        return x1 + x2


class XorAddOperator(Task):
    def operate(self, x1, x2):
        return x1 ^ x2 + x1
