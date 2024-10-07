import _model_simulate_env
def main():
    carla = _model_simulate_env
    carla.main()
    
if __name__ == '__main__':
    try:
        main()
    except:
        KeyboardInterrupt
    finally:
        pass