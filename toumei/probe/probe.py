import torch.nn as nn


def len_str(x):
    if isinstance(x, nn.Sequential):
        return 0
    return len(str(x))


def get_modules(model: nn.Module):
    """
    Returns a dict containing the model named modules
    :param model The inspected model
    :returns dict
    """
    result = []

    for m in model.named_modules():
        result.append(m)

    return dict(result[1:])


def print_modules(model: nn.Module):
    """
    Prints the named modules of a model
    :param model The inspected model
    """
    # get the modules as a dict
    modules = get_modules(model)

    # get the longest key length
    longest_key = max(map(len_str, modules.keys()))

    # get the longest value length
    longest_value = max(map(len_str, modules.values()))

    max_name_length = max(len("Name"), longest_key)
    max_module_length = max(len("Module"), longest_value)

    # print the modules in a table
    print()
    print(f'{"Name" :{" "}<{max_name_length + 2}} | {"Module" :{" "}<{max_module_length + 2}}')
    print('-' * (max_module_length + max_name_length + 5))
    for module in modules:
        if isinstance(modules[module], nn.Sequential):
            continue
        print(f'{module :{" "}<{max_name_length + 2}} | {str(modules[module]) :{" "}<{max_module_length + 2}}')

    print()


