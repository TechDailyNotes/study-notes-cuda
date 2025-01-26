#include <stdio.h>
#include <stdlib.h>

struct Person {
    int age;
    char *name;
};

typedef struct {
    int age;
    char *name;
} Pet;

int main() {
    struct Person person1;
    person1.age = 25;
    person1.name = "Alex";
    printf("person1.age = %d\nperson1.name = %s\n", person1.age, person1.name);

    Pet *pet1 = (Pet*) malloc(sizeof(Pet));
    pet1->age = 1;
    pet1->name = "Bobby";
    printf("pet1->age = %d\npet1->name = %s\n", pet1->age, pet1->name);

    return 0;
}
