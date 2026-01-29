#include <stdio.h> 
#include <assert.h>
#include <stdlib.h>

/*
 * JSON Parser for the keywords JSON config
 * Data structure is key : list in a single JSON object, for n keys
*/

void printCharArr(char arr[]) {
    int i = 0;
    while (arr[i] != '\0') {
        putchar(current_key[i]);
        i++;
    }
    printf("\n");
}

int peekNext(FILE* fptr) {
    // Check what the next character is
    fpos_t starting_pos;
    if (fgetpos(fptr, &starting_pos) != 0) {
        printf("fgetpos failed!\n");
        fclose(fptr);
        exit(1);
    }
  
    // move forward one byte
    if (fseek(fptr, 1, SEEK_CUR) != 0) {
        printf("fseek failed!\n");
        fclose(fptr);
        exit(1);
    }

    int peeked = fgetc(fptr);
    if (fsetpos(fptr, &starting_pos) != 0) {
        printf("fsetpos failed!\n");
        fclose(fptr);
        exit(1);
    }

    return peeked;
}

void getKey(FILE* fptr, char key[], int longest_key) {
    // only called once a " is found and we haven't found a key yet
    // current position of file pointer is the char after "
    int k = 0;
    do {
        if (k >= longest_key) {
            // we only get here if the next char is not : and our key buffer is full
            // i.e, we have found a key that is too long
            printf("Found a key that is too long! Be sure to check all keys!\n");
            fclose(fptr);
            exit(1);
        } 
        key[k] = fgetc(fptr);
        k++; 
    } while (peekNext(fptr) != ':');

    key[k] = '\0';
}

int main(int argc, char* argv[]) {
    // I want to make a hash map from this json and it can be "slow"
    // First go through whole file and get the number of keys to create our hashmap
    // Then go back through the file and get all the lists and place them in their key's hash

    if (argc != 2) { printf("There should be one argument! The path to the json\n"); return 0; }
    const char* path = argv[1];
    int c;

    FILE *fptr;
    fptr = fopen(path, "r");
    (void)assert(fptr != NULL); 
    
    int longest_key = 2;
    char current_key[longest_key + 1] = {};
    while ((c = fgetc(fptr)) != -1) {
        assert(c != '\'');

        if (c == '"') {
            (void)getKey(fptr, current_key, longest_key);
            (void)printCharArr(current_key); 
            return 0;
        }
        
        if (current_key[0] != '\0') {} // this means we are currently inside a key

        // TODO make sure to reset current_key: current_key = {};
    }

    fclose(fptr);
    return 0;
}
