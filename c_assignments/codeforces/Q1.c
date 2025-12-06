#include <stdio.h>

int main(){
    int w;
    scanf("%d", &w);

    if(w <= 100 && w >= 1){
        if(!(w%2==0) || w==2){
            printf("NO");
        }
        else{
            printf("YES");
        }
    }
    else{
        printf("OUT OF RANGE");
    }
}