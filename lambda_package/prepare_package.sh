#!/bin/bash

creat_function(){
aws lambda create-function \
 --function-name deep_speech$i \
 --zip-file fileb://test.zip \
 --handler lambda_function.lambda_handler --runtime python3.7 \
--role arn:aws:iam::494717363231:role/service-role/deep_speech-role-eclmsdrd
}

update_code(){
zip -g deepspeech_package.zip lambda_function.py
aws s3 rm s3://deepspeech-package-new/deepspeech_package.zip
aws s3 cp deepspeech_package.zip s3://deepspeech-package-new
#echo "https://deepspeech-package.s3.amazonaws.com/deepspeech_package.zip"
#aws lambda update-function-code --function-name deep_speech --s3-bucket deepspeech-package-new --s3-key deepspeech_package.zip
}

for i in {0..31..1}
do
	echo $i
	#creat_function	
        aws  lambda update-function-configuration --function-name deep_speech$i --timeout 900
	aws lambda update-function-code --function-name deep_speech$i --s3-bucket deepspeech-package-new --s3-key deepspeech_package.zip

done
