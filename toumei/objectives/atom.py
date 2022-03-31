class Atom(object):
    def __init__(self):
        super(Atom, self).__init__()

    def __str__(self):
        return "Atom()"

    def __call__(self, *args, **kwargs):
        return NotImplementedError

    @property
    def key(self):
        return NotImplementedError

    @property
    def layer(self):
        return NotImplementedError
