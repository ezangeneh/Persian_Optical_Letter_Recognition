#include <QCoreApplication>
#include "farsi_ocr/digitRecognizer.h"
#include "tiny-cnn/tiny_cnn.h"
using namespace tiny_cnn;


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    digitRecognizer dr;

    QString TrainPath = "db_digit_Train";
    QString TestPath = "Digits_DB";
    QDir dir(TestPath);
    QStringList imageNames = dir.entryList(QDir::Filter::Files);

//    dr.TRAIN(cv::Size(32,32),TrainPath);
    dr.FeatureExtractionCNN(cv::Size(32,32),TrainPath);

    int sizeOfTestData = imageNames.size();
    unsigned int countKNN = 0 , countCNN = 0;
    for(int i = 0; i < sizeOfTestData; i++)
    {
        qDebug() << i+1 <<" Of "<<sizeOfTestData;

        QString filename = TestPath + QString("/") + imageNames[i];

        auto label_KNN = dr.testWithKnn(cv::Size(32,32),filename);

        auto label_CNN = dr.test(filename,cv::Size(32,32));

        QChar label_str = filename[filename.size()-5];
        label_t label_true = QString(label_str).toInt();


        if(label_KNN == label_true)
            ++countKNN;

        if(label_CNN == label_true)
            ++countCNN;

    }

    qDebug()<<"Accuracy of KNN = "<<(double)countKNN/sizeOfTestData;
    qDebug()<<"Accuracy of CNN = "<<(double)countCNN/sizeOfTestData;

    return a.exec();
}
