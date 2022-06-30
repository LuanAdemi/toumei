from abc import abstractmethod, ABC


class Task(ABC):
    """An abstract class that handles different tasks.
    Its abstract method is 'operate', which all tasks have to implement."""

    @abstractmethod
    def operate(self, *args):
        pass

    @abstractmethod
    def get_name(self):
        return "task"


class XorOperator(Task):
    def operate(self, x1, x2):
        return x1 ^ x2

    def get_name(self):
        return "xor"


class AddOperator(Task):
    def operate(self, x1, x2):
        return x1 + x2

    def get_name(self):
        return "add"


class XorAddOperator(Task):
    def operate(self, x1, x2):
        return x1 ^ x2 + x1

    def get_name(self):
        return "xoradd"
