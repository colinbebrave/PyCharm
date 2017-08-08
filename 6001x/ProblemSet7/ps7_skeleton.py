import random as rand
import string

class AdoptionCenter:
    """
    The AdoptionCenter class stores the important information that a
    client would need to know about, such as the different numbers of
    species stored, the location, and the name. It also has a method to adopt a pet.
    """
    def __init__(self, name, species_types, location):
        self.name = name
        self.species_types = species_types
        self.location = location
    def get_number_of_species(self, animal):
        if animal in self.species_types.keys():
            return self.species_types[animal]
        else:
            return 0
    def get_location(self):
        self.location = (float(self.location[0]), float(self.location[1]))
        return self.location
    def get_species_count(self):
        return self.species_types.copy()
    def get_name(self):
        return self.name
    def adopt_pet(self, species):
        self.species_types[species] -= 1
        if self.species_types[species] == 0 :
            self.species_types.pop(species,0)


class Adopter:
    """ 
    Adopters represent people interested in adopting a species.
    They have a desired species type that they want, and their score is
    simply the number of species that the shelter has of that species.
    """
    def __init__(self, name, desired_species):
        self.name = name
        self.desired_species = desired_species
    def get_name(self):
        return self.name
    def get_desired_species(self):
        return self.desired_species
    def get_score(self, adoption_center):
        return 1.0 * adoption_center.get_number_of_species(self.get_desired_species())



class FlexibleAdopter(Adopter):
    """
    A FlexibleAdopter still has one type of species that they desire,
    but they are also alright with considering other types of species.
    considered_species is a list containing the other species the adopter will consider
    Their score should be 1x their desired species + .3x all of their desired species
    """
    # Your Code Here, should contain an __init__ and a get_score method.
    def __init_(self,name,desired_species,considered_species):
        Adopter.__init__(name,desired_species)
        self.considered_species = considered_species
    def get_score(self, adoption_center):
        num_other = 0
        for item in self.considered_species:
            num_other += adoption_center.get_number_of_species(item)
        return Adopter.get_score(self,adoption_center) + 0.3 * num_other

class FearfulAdopter(Adopter):
    """
    A FearfulAdopter is afraid of a particular species of animal.
    If the adoption center has one or more of those animals in it, they will
    be a bit more reluctant to go there due to the presence of the feared species.
    Their score should be 1x number of desired species - .3x the number of feared species
    """
    # Your Code Here, should contain an __init__ and a get_score method.
    def __init__(self,name,desired_species,feared_species):
        Adopter.__init__(self,name,desired_species)
        self.feared_species = feared_species
    def get_score(self,adoption_center):
        num_feared = 0
        for item in self.feared_species:
            num_feared += adoption_center.get_number_of_species(item)
        return max(Adopter.get_score(self,adoption_center) - 0.3 * num_feared,0.0)

class AllergicAdopter(Adopter, object):
    """
    An AllergicAdopter is extremely allergic to a one or more species and cannot
    even be around it a little bit! If the adoption center contains one or more of
    these animals, they will not go there.
    Score should be 0 if the center contains any of the animals, or 1x number of desired animals if not
    """

    def __init__(self, name, desired_species, allergic_species):
        Adopter.__init__(self, name, desired_species)
        self.allergic_species = allergic_species

    def get_score(self, adoption_center):
        adopter_score = super(AllergicAdopter, self).get_score(adoption_center)
        makeZero = False
        for animal in self.allergic_species:
            if animal in adoption_center.species_types:
                makeZero = True
        if makeZero:
            final_score = 0
        else:
            final_score = adopter_score
        return float(final_score)


class MedicatedAllergicAdopter(AllergicAdopter):
    """
    A MedicatedAllergicAdopter is extremely allergic to a particular species
    However! They have a medicine of varying effectiveness, which will be given in a dictionary
    To calculate the score for a specific adoption center, we want to find what is the most allergy-inducing species that the adoption center has for the particular MedicatedAllergicAdopter.
    To do this, first examine what species the AdoptionCenter has that the MedicatedAllergicAdopter is allergic to, then compare them to the medicine_effectiveness dictionary.
    Take the lowest medicine_effectiveness found for these species, and multiply that value by the Adopter's calculate score method.
    """

    def __init__(self, name, desired_species, allergic_species, medicine_effectiveness):
        Adopter.__init__(self, name, desired_species)
        self.allergic_species = allergic_species
        self.medicine_effectiveness = medicine_effectiveness

    def get_score(self, adoption_center):
        adopter_score = super(AllergicAdopter, self).get_score(adoption_center)
        makeZero = False
        allergy = 1
        for animal in self.allergic_species:
            if animal in adoption_center.species_types:
                # compare worst case scenario
                if self.medicine_effectiveness[animal] < allergy:
                    # then make that effectiveness the worst possible outcome
                    allergy = self.medicine_effectiveness[animal]
        final_score = adopter_score * allergy
        return float(final_score)


class AllergicAdopter1(Adopter):
    """
    An AllergicAdopter is extremely allergic to a one or more species and cannot
    even be around it a little bit! If the adoption center contains one or more of
    these animals, they will not go there.
    Score should be 0 if the center contains any of the animals, or 1x number of desired animals if not
    """
    # Your Code Here, should contain an __init__ and a get_score method.
    def __init__(self,name,desired_species,allergic_species):
        Adopter.__init__(self,name,desired_species)
        self.allergic_species = allergic_species
    def get_score(self,adoption_center):
        for item in self.allergic_species:
            if AdoptionCenter.get_number_of_species(self,item) > 0 :
                return 0.0
            else:
                return Adopter.get_score(self,adoption_center)

class MedicatedAllergicAdopter1(AllergicAdopter):
    """
    A MedicatedAllergicAdopter is extremely allergic to a particular species
    However! They have a medicine of varying effectiveness, which will be given in a dictionary
    To calculate the score for a specific adoption center, we want to find what is the most allergy-inducing species that the adoption center has for the particular MedicatedAllergicAdopter. 
    To do this, first examine what species the AdoptionCenter has that the MedicatedAllergicAdopter is allergic to, then compare them to the medicine_effectiveness dictionary. 
    Take the lowest medicine_effectiveness found for these species, and multiply that value by the Adopter's calculate score method.
    """
    # Your Code Here, should contain an __init__ and a get_score method.
    def __init__(self, name, desired_species, allergic_species, medicine_effectiveness):
        AllergicAdopter.__init__(self,name,desired_species,allergic_species)
        self.medicine_effectiveness = medicine_effectiveness
    def get_score(self,adoption_center):
        effect = 1
        for key in AdoptionCenter.get_species_count().keys():
            if key in self.medicine_effectiveness.keys():
                if self.medicine_effectiveness[key] < effect:
                    effect = self.medicine_effectiveness[key]
        return effect * Adopter.get_score(self,adoption_center)


class SluggishAdopter(Adopter):
    """
    A SluggishAdopter really dislikes travelleng. The further away the
    AdoptionCenter is linearly, the less likely they will want to visit it.
    Since we are not sure the specific mood the SluggishAdopter will be in on a
    given day, we will asign their score with a random modifier depending on
    distance as a guess.
    Score should be
    If distance < 1 return 1 x number of desired species
    elif distance < 3 return random between (.7, .9) times number of desired species
    elif distance < 5. return random between (.5, .7 times number of desired species
    else return random between (.1, .5) times number of desired species
    """
    # Your Code Here, should contain an __init__ and a get_score method.
    def __init__(self,name,desired_species,location):
        Adopter.__init__(self,name,desired_species)
        self.location = location
    def get_linear_distance(self,to_location):
        return sqrt((to_location[0] - self.location[0])**2 + (to_location[1] - self.location[1])**2)
    def get_score(self, adoption_center):
        the_location = adoption_center.get_location()
        d = self.get_linear_distance(the_location)
        adopter_score = Adopter.get_score(self,adoption_center)
        if d < 1.0 :
            return 1.0 * adopter_score
        elif 1.0 <= d < 3.0 :
            return rand.uniform(0.7,0.9) * adopter_score
        elif 3.0 <= d < 5.0 :
            return rand.uniform(0.5,0.7) * adopter_score
        else:
            return rand.uniform(0.1,0.5) * adopter_score
def get_ordered_adoption_center_list(adopter, list_of_adoption_centers):
    """
    The method returns a list of an organized adoption_center such that the scores for each AdoptionCenter to the Adopter will be ordered from highest score to lowest score.
    """
    # Your Code Here
    score_list = []
    for adoption_center in list_of_adoption_centers:
        score_list.append([adoption_center,adopter.get_score(adoption_center)])
    score_list = sorted(score_list,key = lambda x: x[0].get_name())
    score_list = sorted(score_list,key = lambda x: x[1],reverse = True)
    return [element[0] for element in score_list]


def get_adopters_for_advertisement(adoption_center, list_of_adopters, n):
    """
    The function returns a list of the top n scoring Adopters from list_of_adopters (in numerical order of score)
    """
    # Your Code Here
    score_list = []
    for adopter in list_of_adopters:
        score_list.append([adopter,adopter.get_score(adoption_center)])
    score_list = sorted(score_list,key = lambda x: x[0].get_name())
    score_list = sorted(score_list,key = lambda x: x[1],reverse = True)
    return [x[0] for x in score_list[0:n]]
