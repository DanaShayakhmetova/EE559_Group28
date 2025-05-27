#### Steps to run the chrome extension

##### Make sure before running:

1) Model weights are inside this folder 
    - gpt-v4-small folder (Implicit)
    - gpt2-small-regression-finetuned-fixed-smaller-train (Rasch)
    - gpt2_hate_speech
    - saved_model_stg1_bert

![Folder Organization](folder_org.png)


##### To run:

1) Run the program in terminal/command line : python app.py 
    - leave it running 
2) Upload the whole twitter_hate_extension folder onto chrome extension 
    - [Link](chrome://extensions/)
    - make sure you turned on developer mode 
    - then press on "load unpacked on the upper left"
3) log into twitter and it should be there 


##### NOTE:
- if you reload the folder, make sure to refresh the twitter page.
- not perfect but still a nice viz of the models. 
- Rerun it on the same input to find better explanations!!!