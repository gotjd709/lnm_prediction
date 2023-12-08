from skimage.measure          import regionprops
import numpy                  as np
import cv2


class FeatureExtraction():
    def __init__(self, mask, adjust):
        ##### mask
        self.mask = mask
        self.adjust = adjust

        self.tot_all_mask = self.make_total_mask(self.mask)[0]
        self.tot_tumor1_mask = self.make_total_mask(self.mask)[1]
        self.tot_tumor2_mask = self.make_total_mask(self.mask)[2]
        self.main_all_mask = self.make_main_mask(self.tot_all_mask, 'all')
        self.main_tumor1_mask = self.make_main_mask(self.tot_tumor1_mask, 'tumor1')
        self.main_tumor2_mask = self.make_main_mask(self.tot_tumor2_mask, 'tumor2')
        
        ##### property
        ### tumor ratio
        self.tot_tumor1_ratio = self.get_ratio(self.tot_tumor1_mask, self.tot_tumor2_mask)[0]
        self.tot_tumor2_ratio = self.get_ratio(self.tot_tumor1_mask, self.tot_tumor2_mask)[1]
        self.main_tumor1_ratio = self.get_ratio(self.main_tumor1_mask, self.main_tumor2_mask)[0]
        self.main_tumor2_ratio = self.get_ratio(self.main_tumor1_mask, self.main_tumor2_mask)[1]

        # total region all tumor
        self.tot_all_num_region = self.num_region(self.tot_all_mask)
        self.tot_all_area = self.get_properties('area', self.tot_all_mask)*self.adjust
        self.tot_all_convex_area = self.get_properties('convex_area', self.tot_all_mask)*self.adjust
        self.tot_all_filled_area = self.get_properties('filled_area', self.tot_all_mask)*self.adjust
        self.tot_all_perimeter = self.get_properties('perimeter', self.tot_all_mask)*self.adjust
        self.tot_all_equiv_diameter = self.get_properties('equivalent_diameter', self.tot_all_mask)*self.adjust
        self.tot_all_euler_number = self.get_properties('euler_number', self.tot_all_mask)
        self.tot_all_mj_axis_length = self.get_properties('major_axis_length', self.tot_all_mask)*self.adjust
        self.tot_all_mi_axis_length = self.get_properties('minor_axis_length', self.tot_all_mask)*self.adjust
        # total region tumor1
        self.tot_tumor1_num_region = self.num_region(self.tot_tumor1_mask)
        self.tot_tumor1_area = self.get_properties('area', self.tot_tumor1_mask)*self.adjust
        self.tot_tumor1_convex_area = self.get_properties('convex_area', self.tot_tumor1_mask)*self.adjust
        self.tot_tumor1_filled_area = self.get_properties('filled_area', self.tot_tumor1_mask)*self.adjust
        self.tot_tumor1_perimeter = self.get_properties('perimeter', self.tot_tumor1_mask)*self.adjust
        self.tot_tumor1_equiv_diameter = self.get_properties('equivalent_diameter', self.tot_tumor1_mask)*self.adjust
        self.tot_tumor1_euler_number = self.get_properties('euler_number', self.tot_tumor1_mask)
        self.tot_tumor1_mj_axis_length = self.get_properties('major_axis_length', self.tot_tumor1_mask)*self.adjust
        self.tot_tumor1_mi_axis_length = self.get_properties('minor_axis_length', self.tot_tumor1_mask)*self.adjust
        # total region tumor2
        self.tot_tumor2_num_region = self.num_region(self.tot_tumor2_mask)
        self.tot_tumor2_area = self.get_properties('area', self.tot_tumor2_mask)*self.adjust
        self.tot_tumor2_convex_area = self.get_properties('convex_area', self.tot_tumor2_mask)*self.adjust
        self.tot_tumor2_filled_area = self.get_properties('filled_area', self.tot_tumor2_mask)*self.adjust
        self.tot_tumor2_perimeter = self.get_properties('perimeter', self.tot_tumor2_mask)*self.adjust
        self.tot_tumor2_equiv_diameter = self.get_properties('equivalent_diameter', self.tot_tumor2_mask)*self.adjust
        self.tot_tumor2_euler_number = self.get_properties('euler_number', self.tot_tumor2_mask)
        self.tot_tumor2_mj_axis_length = self.get_properties('major_axis_length', self.tot_tumor2_mask)*self.adjust
        self.tot_tumor2_mi_axis_length = self.get_properties('minor_axis_length', self.tot_tumor2_mask)*self.adjust

        # main region all tumor
        self.main_all_area = self.get_properties('area', self.main_all_mask)*self.adjust
        self.main_all_convex_area = self.get_properties('convex_area', self.main_all_mask)*self.adjust
        self.main_all_filled_area = self.get_properties('filled_area', self.main_all_mask)*self.adjust
        self.main_all_perimeter = self.get_properties('perimeter', self.main_all_mask)*self.adjust
        self.main_all_equiv_diameter = self.get_properties('equivalent_diameter', self.main_all_mask)*self.adjust
        self.main_all_euler_number = self.get_properties('euler_number', self.main_all_mask)
        self.main_all_mj_axis_length = self.get_properties('major_axis_length', self.main_all_mask)*self.adjust
        self.main_all_mi_axis_length = self.get_properties('minor_axis_length', self.main_all_mask)*self.adjust
        self.main_all_eccentricity = self.get_properties('eccentricity', self.main_all_mask)
        self.main_all_extent = self.get_properties('extent', self.main_all_mask)
        self.main_all_solidity = self.get_properties('solidity', self.main_all_mask)
        self.main_all_pa_ratio = self.get_properties('pa_ratio', self.main_all_mask)
        self.main_all_fractal_dimension = self.get_properties('fractal_dimension', self.main_all_mask)
        # main region tumor1
        self.main_tumor1_area = self.get_properties('area', self.main_tumor1_mask)*self.adjust
        self.main_tumor1_convex_area = self.get_properties('convex_area', self.main_tumor1_mask)*self.adjust
        self.main_tumor1_filled_area = self.get_properties('filled_area', self.main_tumor1_mask)*self.adjust
        self.main_tumor1_perimeter = self.get_properties('perimeter', self.main_tumor1_mask)*self.adjust
        self.main_tumor1_equiv_diameter = self.get_properties('equivalent_diameter', self.main_tumor1_mask)*self.adjust
        self.main_tumor1_euler_number = self.get_properties('euler_number', self.main_tumor1_mask)
        self.main_tumor1_mj_axis_length = self.get_properties('major_axis_length', self.main_tumor1_mask)*self.adjust
        self.main_tumor1_mi_axis_length = self.get_properties('minor_axis_length', self.main_tumor1_mask)*self.adjust
        self.main_tumor1_all_eccentricity = self.get_properties('eccentricity', self.main_tumor1_mask)
        self.main_tumor1_all_extent = self.get_properties('extent', self.main_tumor1_mask)
        self.main_tumor1_all_solidity = self.get_properties('solidity', self.main_tumor1_mask)
        self.main_tumor1_all_pa_ratio = self.get_properties('pa_ratio', self.main_tumor1_mask)
        self.main_tumor1_all_fractal_dimension = self.get_properties('fractal_dimension', self.main_tumor1_mask)
        # main region tumor2
        self.main_tumor2_area = self.get_properties('area', self.main_tumor2_mask)*self.adjust
        self.main_tumor2_convex_area = self.get_properties('convex_area', self.main_tumor2_mask)*self.adjust
        self.main_tumor2_filled_area = self.get_properties('filled_area', self.main_tumor2_mask)*self.adjust
        self.main_tumor2_perimeter = self.get_properties('perimeter', self.main_tumor2_mask)*self.adjust
        self.main_tumor2_equiv_diameter = self.get_properties('equivalent_diameter', self.main_tumor2_mask)*self.adjust
        self.main_tumor2_euler_number = self.get_properties('euler_number', self.main_tumor2_mask)
        self.main_tumor2_mj_axis_length = self.get_properties('major_axis_length', self.main_tumor2_mask)*self.adjust
        self.main_tumor2_mi_axis_length = self.get_properties('minor_axis_length', self.main_tumor2_mask)*self.adjust
        self.main_tumor2_eccentricity = self.get_properties('eccentricity', self.main_tumor2_mask)
        self.main_tumor2_extent = self.get_properties('extent', self.main_tumor2_mask)
        self.main_tumor2_solidity = self.get_properties('solidity', self.main_tumor2_mask)
        self.main_tumor2_pa_ratio = self.get_properties('pa_ratio', self.main_tumor2_mask)
        self.main_tumor2_fractal_dimension = self.get_properties('fractal_dimension', self.main_tumor2_mask)

    def make_total_mask(self, mask):
        tumor1_mask = np.where(mask==2,1,0).astype(np.uint8)
        tumor2_mask = np.where(mask==3,1,0).astype(np.uint8)
        all_mask = (tumor1_mask+tumor2_mask).astype(np.uint8)
        return all_mask, tumor1_mask, tumor2_mask

    def make_main_mask(self, mask, tumor_type):
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[1:,-1]
        try:
            # all tumor -> 5, tumor1 -> 3, tumor2 -> 3
            main_threshold = sorted(sizes)[-5] if tumor_type == 'all' else sorted(sizes)[-3]
            main_mask = np.zeros((mask.shape)).astype(np.uint8)      
            for i in range(0, nb_components-1):
                if sizes[i] >= main_threshold:
                    main_mask[output == i+1] = 1 
            return main_mask
        except:
            # all tumor < 5, tumor1 < 3, tumor2 < 3
            return mask

    # get tumor ratio method
    def get_ratio(self, tumor1_mask, tumor2_mask):
        tissue_mask = np.where(self.mask==1,1,0).astype(np.uint8)

        _, counts0 = np.unique(tissue_mask, return_counts = True)
        _, counts1 = np.unique(tumor1_mask, return_counts = True)
        _, counts2 = np.unique(tumor2_mask, return_counts = True)

        tumor1_ratio = counts1[-1]/(counts0[-1]+counts1[-1]+counts2[-1])
        tumor2_ratio = counts2[-1]/(counts0[-1]+counts1[-1]+counts2[-1])
        return round(tumor1_ratio, 4), round(tumor2_ratio, 4)

    def num_region(self, mask):
        nb_components, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return nb_components-1

    # get properties method
    def properties(self, properties, mask):
        prop = sum([region[properties] for region in regionprops(mask)])
        return round(prop, 4)

    def pa_ratio(self, mask):
        prop = sum([region['perimeter'] for region in regionprops(mask)])/sum([region['area'] for region in regionprops(mask)])
        return round(prop, 4)

    def fractal_dimension(self, mask, threshold=0.9):
        nb_components, output, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        prop = 0
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)
            return len(np.where((S > 0) & (S < k*k))[0])
        for i in range(1, nb_components):
            Z = (np.where(output==i,1,0)).astype(np.uint8)
            assert(len(Z.shape) == 2)
            Z = (Z < threshold)
            p = min(Z.shape)
            n = 2**np.floor(np.log(p)/np.log(2))
            n = int(np.log(n)/np.log(2))
            sizes = 2**np.arange(n, 1, -1)
            counts = []
            for size in sizes:
                counts.append(boxcount(Z, size))
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            prop += -coeffs[0]
        prop = prop/(nb_components-1)
        return round(prop, 4)

    def get_properties(self, properties, mask):
        if np.max(mask) == 0:
            return 0
        if properties == 'num_regions':
            return self.num_region(mask)
        elif (properties == 'area') or (properties == 'filled_area') or (properties == 'convex_area') or (properties == 'euler_number') or (properties == 'perimeter') or (properties == 'equivalent_diameter') or (properties == 'major_axis_length') or (properties == 'minor_axis_length') or (properties == 'eccentricity') or (properties == 'extent') or (properties == 'solidity'):
            return self.properties(properties, mask)
        elif properties == 'pa_ratio':
            return self.pa_ratio(mask)
        elif properties == 'fractal_dimension':
            return self.fractal_dimension(mask)