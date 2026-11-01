def build_darkroom_data_filename(env, n_envs, config, mode):
    """
    Builds the filename for the darkroom data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename_template = 'data/darkroom/trajs_{}.pkl'
    filename = env
    filename += '_envs' + str(n_envs)
    if mode != 2:
        filename += '_hists' + str(config['n_hists'])
        filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_mix' + str(config['mix'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        # filename += '_' + config['rollin_type']
        filename += '_eval'
        
    return filename_template.format(filename)

def build_darkroom_model_filename(env, config):
    """
    Builds the filename for the darkroom model.
    """
    filename = f'models/{env}/'
    if config['Q']:
        filename += 'Q'
        if config['soft_max']:
            filename += '_softmax_'
        else:
            filename += '_hardmax_'
            
    filename += 'mix' + str(config['mix'])
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    filename += '_envs' + str(config['n_envs'])
    filename += '_hists' + str(config['n_hists'])
    filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_seed' + str(config['seed'])
    return filename
