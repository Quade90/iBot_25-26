#include <stdio.h>

int main(){
    int nums[] = {3,7,4,9,4,5,7,4,67,4,3,5,7,4,3,5,7,7,3,2,4,54,67,9,9,5,54,3,4,5,7,89,9,5,7,5,4,2,2,4,54,6,88,};
    int k = 36;
    
    int sum = 0;

    for(int i = 0; i<=sizeof(nums)/sizeof(nums[0]); i++){
        sum+=nums[i];
    }

    printf("%d",sum%k);
}