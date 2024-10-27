import argparse

class BaseParser(dict):
    def __init__(self, cmd_args, *file_args):
        super(BaseParser, self).__init__()
        args = [*file_args, cmd_args] # cmd_args has higher priority than file_args

        # default options for the parser, which can be customized for specific applications
        self.choice_dict = {}
        self.default_dict = {}
        self.allowed_overwritten_list = {'seed': None}

        for i_d, d in enumerate(args):
            # print(i_d)
            if isinstance(d, argparse.Namespace):
                d = vars(d)
                
            for k, v in d.items():
                assert k not in self.keys() or k in self.allowed_overwritten_list.keys(), f'duplicated arguments {k}, please check the configuration file.'

                if k in self.allowed_overwritten_list.keys() and v == self.allowed_overwritten_list[k]:
                    continue
                # print(f'\t{k}: {v}')
                self.add_item(k, v)

        # check whether the default options has been in args; otherswise, add it.
        for k in self.default_dict.keys():
            if k not in self.keys():
                self[k] = self.default_dict[k] 

    def add_item(self, k, v):
        # 1. convert '-' to '_'; 2. replace string 'None' with NoneType
        k = k.replace('-', '_')
        
        #check whether arguments match the limited choices
        if k in self.choice_dict.keys() and v not in self.choice_dict[k]:
            raise ValueError(f'Illegal argument \'{k}\' for choices {self.choice_dict[k]}')
        
        # convert string None to Nonetype, which is a side effect of using yaml        
        self[k] = None if v == 'None' else v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f'{name}')

    def __setattr__(self, key, val):
        self[key] = val