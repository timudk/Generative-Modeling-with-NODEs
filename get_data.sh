wget https://zenodo.org/record/1161203/files/data.tar.gz?download=1 -O data.tar.gz &&
tar -zxvf data.tar.gz &&
rm data.tar.gz &&
rm -rf data/gas &&
rm -rf data/hepmass &&
rm -rf data/mnist &&
rm -rf data/cifar10 &&
rm -rf data/BSDS300 &&
mkdir data/Freyfaces &&
wget https://github.com/riannevdberg/sylvester-flows/raw/master/data/Freyfaces/freyfaces.pkl -O data/Freyfaces/freyfaces.pkl 