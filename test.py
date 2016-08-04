import preprocess
import configparser

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    mode = 'local'
    preprocess.prepare_augmented_images(config[mode])

if __name__ == '__main__':
    main()
