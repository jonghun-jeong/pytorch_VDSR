import numpy as np



def KC_Pruning(weights):
    new_weights_sequence=[]
    new_bias_sequence=[]
    Prune_check_sequence=[]
    new_w_sum=0
    new_b_sum=0
    w_sum=0
    b_sum=0
    for lay in range(1,20):
        threshold=0
        c_cnt=0
        if lay!=1:           
            temp_w = np.zeros((weights[lay].shape[0],weights[lay].shape[1]-len(Prune_check_sequence),weights[lay].shape[2],weights[lay].shape[3])) 
            
            for i in range(0,weights[lay].shape[1]):
                if ((i in Prune_check_sequence)==False):
                    temp_w[:,c_cnt,:,:]=weights[lay][:,i,:,:]
                    
                    c_cnt+=1
                else:
                    Prune_check_sequence.remove(i)

        else:
            temp_w=weights[lay]


        if lay == 19:
            break

        kernel_size=temp_w.shape[0]
        normlist=np.zeros(kernel_size)
        for i in range(0,kernel_size):
            temp=abs(temp_w[i,:,:,:])
            normlist[i]=(np.sum(temp**2))**0.5
        std=np.std(normlist)
        avg=np.mean(normlist)
        #print("std: ",std, "avg: ", avg)
        threshold=avg-std*1.1

       
        for i in range(0,kernel_size):
            if(normlist[i]<threshold):
                Prune_check_sequence.append(i)
        #make new weights
        new_weights=np.zeros((temp_w.shape[0]-len(Prune_check_sequence),temp_w.shape[1],temp_w.shape[2],temp_w.shape[3]),dtype='float32')
        new_bias=np.zeros((temp_w.shape[0]-len(Prune_check_sequence)),dtype='float32')
        
        
        
        
        w_sum+=weights[lay].shape[0]*weights[lay].shape[1]*weights[lay].shape[2]*weights[lay].shape[3]
       # b_sum+=bias[lay].shape[0];
        new_w_sum+=new_weights.shape[0]*new_weights.shape[1]*new_weights.shape[2]*new_weights.shape[3]
       # new_b_sum+=new_bias.shape[0]


        cnt=0

        for i in range(0,temp_w.shape[0]):
            if((i in Prune_check_sequence)==False):
                new_weights[cnt,:,:,:]=temp_w[i,:,:,:]
                #new_bias[cnt]=bias[i]
                cnt+=1
        
        new_weights_sequence.append(new_weights)
        new_bias_sequence.append(new_bias)
        print(new_weights.shape)
    print(100.0*new_w_sum/w_sum)
    new_weights_sequence.append(weights[0])
    new_weights_sequence.append(temp_w)
    

    return new_weights_sequence, new_bias_sequence
