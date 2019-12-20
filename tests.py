from main import * 
import datetime

def model_reproducability(data_sz=100, gen_test_samples=False,persist=False):
    # This test will iteratively run through all models and try and run .predict on them 
    # and compare them against 100 locally generated results stored in teh repo
    # The output is expected to be 100% or very high % match to the stored results
    
    _, df_ts=data_loader(data_size=data_sz,)

    print('Testing model reproducable (.pred) against stored results')
    vects= ['bow','tfidf']
    classfrs= ['logistic','GBM']

    for vect in vects:
        for classf in classfrs:
            dt0=datetime.datetime.now()
            sa=SentimentAnalyser(vectorizer=vect,classfier=classf,data_size=data_sz, persist=persist)
            res=sa.predict(list(df_ts.Comment))
            if gen_test_samples:
                with open('./tests/test_results_{0}_{1}.txt'.format(vect,classf),'w') as f:
                    for item in res:
                        f.write("%s\n" % item)
            else:
                with open('./tests/test_results_{0}_{1}.txt'.format(vect,classf),'r') as f:
                    test_values=f.readlines()
                perc_match=100*sum([res[idx]==test_values[idx].replace('\n','') for idx in range(len(res))])/len(res)

                print('{0}% matched for model {1} {2}'.format(int(perc_match),vect,classf))

def test_models_e2e(data_sz=10,persist=False):
    # This test will iteratively run through all models and try and run .train and .eval on them
    # If each can perform this without error then test is successful. Test is run on 10 samples. 
    _, df_ts=data_loader(data_size=data_sz)
    print('Testing .train & .eval')
    vects= ['bow','tfidf']
    classfrs= ['logistic','GBM']

    for vect in vects:
        for classf in classfrs:
            dt0=datetime.datetime.now()
            sa=SentimentAnalyser(vectorizer=vect,classfier=classf,data_size=data_sz, persist=persist)
            sa.train()
            sa.eval(df_ts.Comment,df_ts.Label)
            print('Model {0} {1} can train & evaluate'.format(vect,classf))

if __name__ == "__main__":
    print('=============== TEST 1 ===============')
    model_reproducability(data_sz=100, gen_test_samples=False)
    print('=============== TEST 2 ===============')
    test_models_e2e(data_sz=10)