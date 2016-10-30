#matlab -nosplash -nowindow -r "path='/home/c2tao/phd/GlobalPhone_subset/German/wav/train/';clusterNumber=50;dumpfile='german50'; clusterDetection_commandline" 
ORI=$PWD
echo $PWD
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR
echo $PWD
matlab -nosplash -nowindow -r "path='$1';clusterNumber=$2;dumpfile='$3'; clusterDetection_commandline" 
cd $ORI
echo $PWD
