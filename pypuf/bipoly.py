"""
This module provides a data type that represents polynomials over {-1, 1}: BiPoly
It also provides functions for the generation of commonly used BiPoly instances.
"""
from numpy import bincount, array


def to_dict_notation(mon):
    """
    Converts a list of lists containing indices to the internal representation.
    """
    res = {frozenset(x): 1 for x in mon}
    return res


class BiPoly(object):
    """
    Implements polynomials over {-1, 1}.
    In this class we use a internal representation of monomials:
    {set(indices) : coefficients}
    This means that {set(1,2,3) : 2, set(0) : 5} is interpreted as:
    2*X1*X2*X3 + 5*X0
    Each set in this dict is considered a monomial.

    Each variable of a monomials is in {-1, 1}, this means that X**2 = 1, hence we can
    eliminate these terms and shorten the monomials.
    """

    def __init__(self, monomials=None):
        """
        :param monomials: list or dict
                          Initialize a BiPoly with list or dict of indices.
        """
        if monomials is None:
            self.monomials = {}
            return
        if isinstance(monomials, dict):
            self.monomials = monomials
            return
        if isinstance(monomials, list):
            self.monomials = to_dict_notation(monomials)
            return
        raise Exception("Invalid argument.")

    # -------------------------- Container Type Methods --------------------------

    def __len__(self):
        """
        Returns length of polynomial (number of sums).

        >>> len(BiPoly([[], [1], [2]]))
        3
        """
        return self.monomials.__len__()

    def __getitem__(self, mon):
        """
        Returns coefficient of a monomial.
        :param mon: frozenset
                    Monomial as a frozenset of indices.

        >>> BiPoly({frozenset({1, 2, 3}): 3.141})[frozenset({1, 2, 3})]
        3.141
        """
        assert isinstance(mon, frozenset)
        return self.monomials.__getitem__(mon)

    def __setitem__(self, monomial, coefficient):
        """
        Sets coefficient of a monomial.
        :param monomial: frozenset
                    Monomial as a frozenset of indices.
        :param coefficient: int
                      Coefficient of monomial.
        """
        assert isinstance(monomial, frozenset)
        assert isinstance(coefficient, (int, float))
        return self.monomials.__setitem__(monomial, coefficient)

    def __delitem__(self, monomial):
        """
        Deletes a monomial-coefficient pair from the BiPoly.
        :param mon: frozenset
                    Monomial to be deleted as a frozenset of indices.

        >>> p = BiPoly.linear(4)
        >>> del p[frozenset({2})]
        >>> str(p)
        '  1x₀ +   1x₁ +   1x₃'
        """
        assert isinstance(monomial, frozenset)
        return self.monomials.__delitem__(monomial)

    def __iter__(self):
        """
        Return Iterable of this BiPoly.
        Items are pairs of (monomial, coefficient).
        """
        return self.monomials.items().__iter__()

    def __contains__(self, monomial):
        """
        Returns whether a monomial is contained in the BiPoly
        :param mon: frozenset
                    Monomial as a frozenset of indices.

        >>> frozenset({64}) in BiPoly.linear(128)
        True
        """
        assert isinstance(monomial, frozenset)
        return self.monomials.__contains__(monomial)

    def get(self, monomial):
        """
        Wrapper around dict.get.
        :param monomial: frozenset
                    Monomial as a frozenset of indices.
        """
        assert isinstance(monomial, frozenset)
        return self.monomials.get(monomial)

    # -------------------------- Numeric Type Methods --------------------------

    def __add__(self, other):
        """
        Returns the sum of this BiPoly and other BiPoly.
        :param other: BiPoly
                      BiPoly that is added to self.
        """
        res = self.copy()
        for m, c in other:
            res[m] = (res.get(m) or 0) + c
        return res

    def __sub__(self, other):
        """
        Returns the difference of this BiPoly and other BiPoly.
        :param other: BiPoly
                      BiPoly that is subtracted from self.

        >>> p = BiPoly({frozenset(): 3.141})
        >>> q = BiPoly({frozenset({1}): 1})
        >>> str(p - q)
        '3.141 +  -1x₁'
        """
        return self + other.__neg__()

    def __mul__(self, other):
        """
        Returns the product of this BiPoly and other BiPoly.
        :param other: BiPoly
                      BiPoly that is multiplied to self.

        >>> BiPoly([[1]]) * BiPoly([[1, 2], [3]])
          1x₂ +   1x₁x₃
        """
        res = BiPoly()
        for m1, c1 in self:
            for m2, c2 in other:
                m = m1.symmetric_difference(m2)
                c = (res.get(m) or 0) + c1 * c2
                res[m] = c
        return res

    def __neg__(self):
        """
        Returns the negation of this BiPoly (i.e. 0 - self).
        """
        res = self.copy()
        for m, _ in res:
            res[m] = - res[m]
        return res

    def __str__(self):
        """
        Returns a string that represents this polynomial.

        >>> str(BiPoly([[1]]))
        '  1x₁'

        >>> str(BiPoly([[1, 2, 3, 4]]))
        '  1x₁x₂x₃x₄'

        >>> str(BiPoly([[1], [2], [5], [1, 2, 3, 7]]))
        '  1x₁ +   1x₂ +   1x₅ +   1x₁x₂x₃x₇'
        """
        return ' + '.join([
            f'{val:3}' +
            ''.join([
                'x' + ''.join([chr(0x2080 + int(digit)) for digit in str(idx)])
                for idx in m
            ])
            for m, val in self
        ])

    def __repr__(self):
        return self.__str__()

    def __pow__(self, power, modulo=None):
        """
        Returns the exponentiation of this polynomial to the power of integer power.
        :param power: int
                  Power to which this BiPoly will be raised.

        >>> p = BiPoly([[1], [2]])**2
        >>> str(p)
        '  2 +   2x₁x₂'

        >>> str(p**2)
        '  8 +   8x₁x₂'

        >>> str(p**3)
        ' 32 +  32x₁x₂'
        """
        if modulo:
            raise ValueError('Modulo exponentiation not supported by %s' % self.__class__.__name__)
        if power == 1:
            return self.copy()
        # k is not computed yet -> split in two halves
        k_star = power // 2
        self_pow_k_star = self**k_star
        res = self_pow_k_star * self_pow_k_star
        # If k was uneven, we need to multiply with the base poly again
        if power % 2 == 1:
            res = res * self
        return res

    # -------------------------- Custom Methods --------------------------

    def copy(self):
        """
        Returns a copy of this BiPoly.
        """
        return BiPoly(self.monomials.copy())

    def deg(self):
        """
        Returns the degree of this BiPoly.

        >>> BiPoly.linear(42).deg()
        1
        >>> BiPoly.arbiter_puf(42).deg()
        42
        """
        return max(self.degrees())

    def degrees(self):
        """
        Returns an array of all degrees of this BiPoly, and how often they occur,
        for example:

        >>> p = BiPoly([[1],[1,2],[3,4]])
        >>> str(p)
        '  1x₁ +   1x₁x₂ +   1x₃x₄'
        >>> p.degrees()
        array([1, 2, 2])
        """
        return array(list(map(len, list(self.monomials.keys()))))

    def degrees_count(self):
        """
        Returns an array of degree-occurances, where value v of index i means
        that the degree i occurs v times.

        >>> BiPoly([[1], [2], [5], [1001]]).degrees_count()
        array([0, 4])
        """
        return bincount(self.degrees())

    def low_degrees(self, up_to_degree):
        """
        Returns a copy of this BiPoly, that contains only monomials with degree smaller
        than up_to_degree.
        :param up_to_degree: int
                             Only monomials smaller than up_to_degree will be returned.

        >>> p = BiPoly([[1, 2, 3]])
        >>> str(p.low_degrees(3))
        ''

        >>> p = BiPoly([[1, 2, 3], [1], [], [2, 3]])
        >>> str(p.low_degrees(3))
        '  1x₁ +   1 +   1x₂x₃'
        """
        return BiPoly({
            m: c
            for m, c in self.monomials.items()
            if len(m) < up_to_degree
        })

    def to_index_notation(self):
        """
        Returns list of lists, where each list represents a monomial and its entries
        are the indices. Note that coefficients will be omitted.

        >>> p = BiPoly([[], [1], [2], [3], [5, 6]])
        >>> p.to_index_notation()
        [[], [1], [2], [3], [5, 6]]
        """
        return [list(s) for s, _ in self]

    def substitute(self, mapping):
        """
        Returns BiPoly that corresponds to substituting each variable by a monomial
        defined in mapping.
        :param mapping: list of lists
                        list of lists where indices in list i with replace Xi in self.

        >>> atf = BiPoly({frozenset({1, 2, 3, 4}): 1, frozenset({2, 3, 4}): 2, frozenset({3, 4}): 3, frozenset({4}): 4})
        >>> ltf = atf.substitute([[0], [1, 2], [2, 3], [3, 4], [4]])
        >>> str(ltf)
        '  1x₁ +   2x₂ +   3x₃ +   4x₄'

        >>> ltf = BiPoly({frozenset({0}): 0, frozenset({1}): 1, frozenset({2}): 2, frozenset({3}): 3})
        >>> atf = ltf.substitute(([[0, 1, 2, 3], [1, 2, 3], [2, 3], [3]]))
        >>> str(atf)
        '  0x₀x₁x₂x₃ +   1x₁x₂x₃ +   2x₂x₃ +   3x₃'
        """
        # For each monomial in self, substitute the entry i with monomial i of mapping
        new_p = BiPoly()
        for m, c in self:
            new_m = frozenset()
            for idx in m:
                new_m = new_m.symmetric_difference(frozenset(mapping[idx]))
            new_p[new_m] = (new_p.get(new_m) or 0) + c
        return new_p

    # -------------------------- Commonly Used Instances --------------------------

    @classmethod
    def linear(cls, n):
        """
        Returns BiPoly where each variable occurs in its identity, i.e. x1+x2+x3+...
        :param n: int
                  Length of the BiPoly.

        >>> str(BiPoly.linear(5))
        '  1x₀ +   1x₁ +   1x₂ +   1x₃ +   1x₄'
        """
        return BiPoly([[i] for i in range(n)])

    @classmethod
    def arbiter_puf(cls, n):
        """
        Returns a BiPoly that corresponds to the usual model of an Arbiter PUF.
        :param n: int
                  Length of the Arbiter PUF.

        >>> str(BiPoly.arbiter_puf(7))
        '  1x₀x₁x₂x₃x₄x₅x₆ +   1x₁x₂x₃x₄x₅x₆ +   1x₂x₃x₄x₅x₆ +   1x₃x₄x₅x₆ +   1x₄x₅x₆ +   1x₅x₆ +   1x₆'
        """
        return BiPoly([list(range(i, n)) for i in range(n)])

    @classmethod
    def xor_arbiter_puf(cls, n, k):
        """
        Returns the linearized BiPoly that corresponds to a n-bit k-XOR Arbiter PUF.
        :param n: int
                  Length of the Arbiter PUFs.
        :param k: int
                  Amount of Arbiter PUFs which will be XOR'd.

        >>> str(BiPoly.xor_arbiter_puf(64, 1)) == str(BiPoly.arbiter_puf(64))
        True
        >>> str(BiPoly.xor_arbiter_puf(3, 1))
        '  1x₀x₁x₂ +   1x₁x₂ +   1x₂'
        >>> str(BiPoly.xor_arbiter_puf(3, 2))
        '  3 +   2x₀ +   2x₀x₁ +   2x₁'
        """
        return cls.arbiter_puf(n)**k

    @classmethod
    def interpose_puf_approximation(cls, n, k_up, k_down, p_up=None):
        """
        Returns a polynomial whose sign approximates the behavior of an Interpose PUF with the given geometry.
        The upper block of the Interpose PUF is approximated by not taking the sign function. Use `p_up` to
        approximate with a different polynomial.

        :param n: int
                  Length of the Arbiter PUFs.
        :param k_up: int
                  Amount of Arbiter PUFs in the upper block.
        :param k_down: int
                  Amount of Arbiter PUFs in the lower block.
        :param p_up: int
                  Optional precomputed BiPoly approximating the upper block.

        >>> str(BiPoly.interpose_puf_approximation(n=3, k_up=1, k_down=1))
        '  1 +   1x₀ +   1x₀x₁ +   1x₀x₁x₂ +   1x₁x₂ +   2x₂'

        >>> str(BiPoly.interpose_puf_approximation(n=3, k_up=1, k_down=2))
        '  9 +   4x₀ +   6x₀x₁ +   8x₀x₁x₂ +   4x₁x₂ +   6x₂ +   6x₁ +   6x₀x₂'
        """
        p_up = p_up or cls.xor_arbiter_puf(n, k_up)

        group_1 = BiPoly()
        for i in range(n//2):
            group_1 = group_1 + (BiPoly([list(range(i, n))]) * p_up)
        group_2 = p_up.copy()
        group_3 = BiPoly([
            list(range(i-1, n))
            for i in range(n//2+2, n+1)
        ])

        return (group_1 + group_2 + group_3)**k_down
