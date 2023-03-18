{-# LANGUAGE CPP #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE PatternGuards #-}
#ifdef __GLASGOW_HASKELL__
{-# LANGUAGE DeriveLift #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
#endif
#if !defined(TESTING) && defined(__GLASGOW_HASKELL__)
{-# LANGUAGE Trustworthy #-}
#endif

{-# OPTIONS_HADDOCK not-home #-}
{-# OPTIONS_GHC -fno-warn-incomplete-uni-patterns #-}

#include "containers.h"

-----------------------------------------------------------------------------
-- |
-- Module      :  Data.IntMap.Internal
-- Copyright   :  (c) Daan Leijen 2002
--                (c) Andriy Palamarchuk 2008
--                (c) wren romano 2016
-- License     :  BSD-style
-- Maintainer  :  libraries@haskell.org
-- Portability :  portable
--
-- = WARNING
--
-- This module is considered __internal__.
--
-- The Package Versioning Policy __does not apply__.
--
-- The contents of this module may change __in any way whatsoever__
-- and __without any warning__ between minor versions of this package.
--
-- Authors importing this module are expected to track development
-- closely.
--
-- = Description
--
-- This defines the data structures and core (hidden) manipulations
-- on representations.
--
-- @since 0.5.9
-----------------------------------------------------------------------------

-- [Note: INLINE bit fiddling]
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- It is essential that the bit fiddling functions like mask, zero, branchMask
-- etc are inlined. If they do not, the memory allocation skyrockets. The GHC
-- usually gets it right, but it is disastrous if it does not. Therefore we
-- explicitly mark these functions INLINE.


-- [Note: Local 'go' functions and capturing]
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- Care must be taken when using 'go' function which captures an argument.
-- Sometimes (for example when the argument is passed to a data constructor,
-- as in insert), GHC heap-allocates more than necessary. Therefore C-- code
-- must be checked for increased allocation when creating and modifying such
-- functions.


-- [Note: Order of constructors]
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- The order of constructors of IntMap matters when considering performance.
-- Currently in GHC 7.0, when type has 3 constructors, they are matched from
-- the first to the last -- the best performance is achieved when the
-- constructors are ordered by frequency.
-- On GHC 7.0, reordering constructors from Nil | Tip | Bin to Bin | Tip | Nil
-- improves the benchmark by circa 10%.
--

module Data.IntMap.Internal (
    -- * Map type
      IntMap(..), Key          -- instance Eq,Show

    -- * Operators
    , (!), (!?), (\\)

    -- * Query
    , null
    , size
    , member
    , notMember
    , lookup
    , findWithDefault
    , lookupLT
    , lookupGT
    , lookupLE
    , lookupGE
    , disjoint

    -- * Construction
    , empty
    , singleton

    -- ** Insertion
    , insert
    , insertWith
    , insertWithKey
    , insertLookupWithKey

    -- ** Delete\/Update
    , delete
    , adjust
    , adjustWithKey
    , update
    , updateWithKey
    , updateLookupWithKey
    , alter
    , alterF

    -- * Combine

    -- ** Union
    , union
    , unionWith
    , unionWithKey
    , unions
    , unionsWith

    -- ** Difference
    , difference
    , differenceWith
    , differenceWithKey

    -- ** Intersection
    , intersection
    , intersectionWith
    , intersectionWithKey

    -- ** Compose
    , compose

    -- ** General combining function
    , SimpleWhenMissing
    , SimpleWhenMatched
    , runWhenMatched
    , runWhenMissing
    , merge
    -- *** @WhenMatched@ tactics
    , zipWithMaybeMatched
    , zipWithMatched
    -- *** @WhenMissing@ tactics
    , mapMaybeMissing
    , dropMissing
    , preserveMissing
    , mapMissing
    , filterMissing

    -- ** Applicative general combining function
    , WhenMissing (..)
    , WhenMatched (..)
    , mergeA
    -- *** @WhenMatched@ tactics
    -- | The tactics described for 'merge' work for
    -- 'mergeA' as well. Furthermore, the following
    -- are available.
    , zipWithMaybeAMatched
    , zipWithAMatched
    -- *** @WhenMissing@ tactics
    -- | The tactics described for 'merge' work for
    -- 'mergeA' as well. Furthermore, the following
    -- are available.
    , traverseMaybeMissing
    , traverseMissing
    , filterAMissing

    -- ** Deprecated general combining function
    , mergeWithKey
    , mergeWithKey'

    -- * Traversal
    -- ** Map
    , map
    , mapWithKey
    , traverseWithKey
    , traverseMaybeWithKey
    , mapAccum
    , mapAccumWithKey
    , mapAccumRWithKey
    , mapKeys
    , mapKeysWith
    , mapKeysMonotonic

    -- * Folds
    , foldr
    , foldl
    , foldrWithKey
    , foldlWithKey
    , foldMapWithKey

    -- ** Strict folds
    , foldr'
    , foldl'
    , foldrWithKey'
    , foldlWithKey'

    -- * Conversion
    , elems
    , keys
    , assocs
    , keysSet
    , fromSet

    -- ** Lists
    , toList
    , fromList
    , fromListWith
    , fromListWithKey

    -- ** Ordered lists
    , toAscList
    , toDescList
    , fromAscList
    , fromAscListWith
    , fromAscListWithKey
    , fromDistinctAscList

    -- * Filter
    , filter
    , filterWithKey
    , restrictKeys
    , withoutKeys
    , partition
    , partitionWithKey

    , takeWhileAntitone
    , dropWhileAntitone
    , spanAntitone

    , mapMaybe
    , mapMaybeWithKey
    , mapEither
    , mapEitherWithKey

    , split
    , splitLookup
    , splitRoot

    -- * Submap
    , isSubmapOf, isSubmapOfBy
    , isProperSubmapOf, isProperSubmapOfBy

    -- * Min\/Max
    , lookupMin
    , lookupMax
    , findMin
    , findMax
    , deleteMin
    , deleteMax
    , deleteFindMin
    , deleteFindMax
    , updateMin
    , updateMax
    , updateMinWithKey
    , updateMaxWithKey
    , minView
    , maxView
    , minViewWithKey
    , maxViewWithKey

    -- * Debugging
    , showTree
    , showTreeWith

    -- * Internal types
    , Mask, Prefix, Nat

    -- * Utility
    , natFromInt
    , intFromNat
    , NE.link
    , NE.linkWithMask
    , bin
    , binCheckLeft
    , binCheckRight
    , zero
    , nomatch
    , match
    , mask
    , maskW
    , shorter
    , branchMask
    , highestBitMask

    -- * Used by "IntMap.Merge.Lazy" and "IntMap.Merge.Strict"
    , mapWhenMissing
    , mapWhenMatched
    , lmapWhenMissing
    , contramapFirstWhenMatched
    , contramapSecondWhenMatched
    , mapGentlyWhenMissing
    , mapGentlyWhenMatched
    ) where

import Data.Functor.Identity (Identity (..))
import Data.Semigroup (Semigroup(stimes))
#if !(MIN_VERSION_base(4,11,0))
import Data.Semigroup (Semigroup((<>)))
#endif
import Data.Semigroup (stimesIdempotentMonoid)
import Data.Functor.Classes

import Control.DeepSeq (NFData(rnf))
import Data.Bits
import qualified Data.Foldable as Foldable
import Data.Maybe (fromMaybe)
import Utils.Containers.Internal.Prelude hiding
  (lookup, map, filter, foldr, foldl, null)
import Prelude ()

import Data.IntMap.NonEmpty.Internal
  ( Prefix, Mask, Distinct(..) )
import qualified Data.IntMap.NonEmpty.Internal as NE
import Data.IntSet.Internal (Key)
import qualified Data.IntSet.Internal as IntSet
import Utils.Containers.Internal.BitUtil
import Utils.Containers.Internal.StrictPair

#ifdef __GLASGOW_HASKELL__
import Data.Coerce
import Data.Data (Data(..), Constr, mkConstr, constrIndex, Fixity(Prefix),
                  DataType, mkDataType, gcast1)
import GHC.Exts (build)
import qualified GHC.Exts as GHCExts
import Text.Read
import Language.Haskell.TH.Syntax (Lift)
-- See Note [ Template Haskell Dependencies ]
import Language.Haskell.TH ()
#endif
import qualified Control.Category as Category


-- A "Nat" is a natural machine word (an unsigned Int)
type Nat = Word

natFromInt :: Key -> Nat
natFromInt = fromIntegral
{-# INLINE natFromInt #-}

intFromNat :: Nat -> Key
intFromNat = fromIntegral
{-# INLINE intFromNat #-}

{--------------------------------------------------------------------
  Types
--------------------------------------------------------------------}

-- | A map of integers to values @a@.

-- See Note: Order of constructors
data IntMap a = NonEmpty !(NE.IntMap a)
              | Nil

-- Some stuff from "Data.IntSet.Internal", for 'restrictKeys' and
-- 'withoutKeys' to use.
type IntSetPrefix = Int
type IntSetBitMap = Word

-- | @since 0.6.6
deriving instance Lift a => Lift (IntMap a)

bitmapOf :: Int -> IntSetBitMap
bitmapOf x = shiftLL 1 (x .&. IntSet.suffixBitMask)
{-# INLINE bitmapOf #-}

{--------------------------------------------------------------------
  Operators
--------------------------------------------------------------------}

-- | \(O(\min(n,W))\). Find the value at a key.
-- Calls 'error' when the element can not be found.
--
-- > fromList [(5,'a'), (3,'b')] ! 1    Error: element not in the map
-- > fromList [(5,'a'), (3,'b')] ! 5 == 'a'

(!) :: IntMap a -> Key -> a
(!) m k = find k m

-- | \(O(\min(n,W))\). Find the value at a key.
-- Returns 'Nothing' when the element can not be found.
--
-- > fromList [(5,'a'), (3,'b')] !? 1 == Nothing
-- > fromList [(5,'a'), (3,'b')] !? 5 == Just 'a'
--
-- @since 0.5.11

(!?) :: IntMap a -> Key -> Maybe a
(!?) m k = lookup k m

-- | Same as 'difference'.
(\\) :: IntMap a -> IntMap b -> IntMap a
m1 \\ m2 = difference m1 m2

infixl 9 !?,\\{-This comment teaches CPP correct behaviour -}

{--------------------------------------------------------------------
  Types
--------------------------------------------------------------------}

instance Monoid (IntMap a) where
    mempty  = empty
    mconcat = unions
    mappend = (<>)

-- | @since 0.5.7
instance Semigroup (IntMap a) where
    (<>)    = union
    stimes  = stimesIdempotentMonoid

-- | Folds in order of increasing key.
instance Foldable.Foldable IntMap where
  fold (NonEmpty m) = Foldable.fold m
  fold Nil = mempty
  {-# INLINABLE fold #-}
  foldr = foldr
  {-# INLINE foldr #-}
  foldl = foldl
  {-# INLINE foldl #-}
  foldMap f (NonEmpty m) = foldMap f m
  foldMap _ Nil = mempty
  {-# INLINE foldMap #-}
  foldl' = foldl'
  {-# INLINE foldl' #-}
  foldr' = foldr'
  {-# INLINE foldr' #-}
  length = size
  {-# INLINE length #-}
  null   = null
  {-# INLINE null #-}
  toList = elems -- NB: Foldable.toList /= IntMap.toList
  {-# INLINE toList #-}
  elem !x (NonEmpty m) = elem x m
  elem !_ Nil = False
  {-# INLINABLE elem #-}
  maximum (NonEmpty m) = maximum m
  maximum Nil = error "Data.Foldable.maximum (for Data.IntMap): empty map"
  {-# INLINABLE maximum #-}
  minimum (NonEmpty m) = minimum m
  minimum Nil = error "Data.Foldable.minimum (for Data.IntMap): empty map"
  {-# INLINABLE minimum #-}
  sum = foldl' (+) 0
  {-# INLINABLE sum #-}
  product = foldl' (*) 1
  {-# INLINABLE product #-}

-- | Traverses in order of increasing key.
instance Traversable IntMap where
    traverse f = traverseWithKey (\_ -> f)
    {-# INLINE traverse #-}
instance NFData a => NFData (IntMap a) where
    rnf (NonEmpty m) = rnf m
    rnf Nil = ()


#if __GLASGOW_HASKELL__

{--------------------------------------------------------------------
  A Data instance
--------------------------------------------------------------------}

-- This instance preserves data abstraction at the cost of inefficiency.
-- We provide limited reflection services for the sake of data abstraction.

instance Data a => Data (IntMap a) where
  gfoldl f z im = z fromList `f` (toList im)
  toConstr _     = fromListConstr
  gunfold k z c  = case constrIndex c of
    1 -> k (z fromList)
    _ -> error "gunfold"
  dataTypeOf _   = intMapDataType
  dataCast1 f    = gcast1 f

fromListConstr :: Constr
fromListConstr = mkConstr intMapDataType "fromList" [] Prefix

intMapDataType :: DataType
intMapDataType = mkDataType "Data.IntMap.Internal.IntMap" [fromListConstr]

#endif

{--------------------------------------------------------------------
  Query
--------------------------------------------------------------------}
-- | \(O(1)\). Is the map empty?
--
-- > Data.IntMap.null (empty)           == True
-- > Data.IntMap.null (singleton 1 'a') == False

null :: IntMap a -> Bool
null (NonEmpty _) = False
null _   = True
{-# INLINE null #-}

-- | \(O(n)\). Number of elements in the map.
--
-- > size empty                                   == 0
-- > size (singleton 1 'a')                       == 1
-- > size (fromList([(1,'a'), (2,'c'), (3,'b')])) == 3
size :: IntMap a -> Int
size (NonEmpty m) = NE.size m
size Nil = 0

-- | \(O(\min(n,W))\). Is the key a member of the map?
--
-- > member 5 (fromList [(5,'a'), (3,'b')]) == True
-- > member 1 (fromList [(5,'a'), (3,'b')]) == False

-- See Note: Local 'go' functions and capturing]
member :: Key -> IntMap a -> Bool
member !k (NonEmpty m) = NE.member k m
member _ Nil = False

-- | \(O(\min(n,W))\). Is the key not a member of the map?
--
-- > notMember 5 (fromList [(5,'a'), (3,'b')]) == False
-- > notMember 1 (fromList [(5,'a'), (3,'b')]) == True

notMember :: Key -> IntMap a -> Bool
notMember k m = not $ member k m

-- | \(O(\min(n,W))\). Lookup the value at a key in the map. See also 'Data.Map.lookup'.

lookup :: Key -> IntMap a -> Maybe a
lookup !k (NonEmpty m) = NE.lookup k m
lookup _ Nil = Nothing


find :: Key -> IntMap a -> a
find !k (NonEmpty m) = NE.find k m
find _ Nil = error "IntMap.!: empty map"

-- | \(O(\min(n,W))\). The expression @('findWithDefault' def k map)@
-- returns the value at key @k@ or returns @def@ when the key is not an
-- element of the map.
--
-- > findWithDefault 'x' 1 (fromList [(5,'a'), (3,'b')]) == 'x'
-- > findWithDefault 'x' 5 (fromList [(5,'a'), (3,'b')]) == 'a'
findWithDefault :: a -> Key -> IntMap a -> a
findWithDefault def !k (NonEmpty m) = NE.findWithDefault def k m
findWithDefault def _ Nil = def


-- | \(O(\min(n,W))\). Find largest key smaller than the given one and return the
-- corresponding (key, value) pair.
--
-- > lookupLT 3 (fromList [(3,'a'), (5,'b')]) == Nothing
-- > lookupLT 4 (fromList [(3,'a'), (5,'b')]) == Just (3, 'a')
lookupLT :: Key -> IntMap a -> Maybe (Key, a)
lookupLT !k (NonEmpty m) = NE.lookupLT k m
lookupLT _ Nil = Nothing


-- | \(O(\min(n,W))\). Find smallest key greater than the given one and return the
-- corresponding (key, value) pair.
--
-- > lookupGT 4 (fromList [(3,'a'), (5,'b')]) == Just (5, 'b')
-- > lookupGT 5 (fromList [(3,'a'), (5,'b')]) == Nothing

lookupGT :: Key -> IntMap a -> Maybe (Key, a)
lookupGT !k (NonEmpty m) = NE.lookupGT k m
lookupGT _ Nil = Nothing


-- | \(O(\min(n,W))\). Find largest key smaller or equal to the given one and return
-- the corresponding (key, value) pair.
--
-- > lookupLE 2 (fromList [(3,'a'), (5,'b')]) == Nothing
-- > lookupLE 4 (fromList [(3,'a'), (5,'b')]) == Just (3, 'a')
-- > lookupLE 5 (fromList [(3,'a'), (5,'b')]) == Just (5, 'b')
lookupLE :: Key -> IntMap a -> Maybe (Key, a)
lookupLE !k (NonEmpty m) = NE.lookupLE k m
lookupLE _ Nil = Nothing

-- | \(O(\min(n,W))\). Find smallest key greater or equal to the given one and return
-- the corresponding (key, value) pair.
--
-- > lookupGE 3 (fromList [(3,'a'), (5,'b')]) == Just (3, 'a')
-- > lookupGE 4 (fromList [(3,'a'), (5,'b')]) == Just (5, 'b')
-- > lookupGE 6 (fromList [(3,'a'), (5,'b')]) == Nothing

-- See Note: Local 'go' functions and capturing.
lookupGE :: Key -> IntMap a -> Maybe (Key, a)
lookupGE !k (NonEmpty m) = NE.lookupGE k m
lookupGE _ Nil = Nothing

{--------------------------------------------------------------------
  Disjoint
--------------------------------------------------------------------}
-- | \(O(n+m)\). Check whether the key sets of two maps are disjoint
-- (i.e. their 'intersection' is empty).
--
-- > disjoint (fromList [(2,'a')]) (fromList [(1,()), (3,())])   == True
-- > disjoint (fromList [(2,'a')]) (fromList [(1,'a'), (2,'b')]) == False
-- > disjoint (fromList [])        (fromList [])                 == True
--
-- > disjoint a b == null (intersection a b)
--
-- @since 0.6.2.1
disjoint :: IntMap a -> IntMap b -> Bool
disjoint Nil _ = True
disjoint _ Nil = True
disjoint (NonEmpty m1) (NonEmpty m2) = NE.disjoint m1 m2

{--------------------------------------------------------------------
  Compose
--------------------------------------------------------------------}
-- | Relate the keys of one map to the values of
-- the other, by using the values of the former as keys for lookups
-- in the latter.
--
-- Complexity: \( O(n * \min(m,W)) \), where \(m\) is the size of the first argument
--
-- > compose (fromList [('a', "A"), ('b', "B")]) (fromList [(1,'a'),(2,'b'),(3,'z')]) = fromList [(1,"A"),(2,"B")]
--
-- @
-- ('compose' bc ab '!?') = (bc '!?') <=< (ab '!?')
-- @
--
-- __Note:__ Prior to v0.6.4, "Data.IntMap.Strict" exposed a version of
-- 'compose' that forced the values of the output 'IntMap'. This version does
-- not force these values.
--
-- @since 0.6.3.1
compose :: IntMap c -> IntMap Int -> IntMap c
compose bc !ab
  | null bc = empty
  | otherwise = mapMaybe (bc !?) ab

{--------------------------------------------------------------------
  Construction
--------------------------------------------------------------------}
-- | \(O(1)\). The empty map.
--
-- > empty      == fromList []
-- > size empty == 0

empty :: IntMap a
empty
  = Nil
{-# INLINE empty #-}

-- | \(O(1)\). A map of one element.
--
-- > singleton 1 'a'        == fromList [(1, 'a')]
-- > size (singleton 1 'a') == 1

singleton :: Key -> a -> IntMap a
singleton k x
  = NonEmpty (NE.Tip k x)
{-# INLINE singleton #-}

{--------------------------------------------------------------------
  Insert
--------------------------------------------------------------------}
-- | \(O(\min(n,W))\). Insert a new key\/value pair in the map.
-- If the key is already present in the map, the associated value is
-- replaced with the supplied value, i.e. 'insert' is equivalent to
-- @'insertWith' 'const'@.
--
-- > insert 5 'x' (fromList [(5,'a'), (3,'b')]) == fromList [(3, 'b'), (5, 'x')]
-- > insert 7 'x' (fromList [(5,'a'), (3,'b')]) == fromList [(3, 'b'), (5, 'a'), (7, 'x')]
-- > insert 5 'x' empty                         == singleton 5 'x'

insert :: Key -> a -> IntMap a -> IntMap a
insert !k x (NonEmpty m) = NonEmpty (NE.insert k x m)
insert k x Nil = NonEmpty (NE.Tip k x)

-- right-biased insertion, used by 'union'
-- | \(O(\min(n,W))\). Insert with a combining function.
-- @'insertWith' f key value mp@
-- will insert the pair (key, value) into @mp@ if key does
-- not exist in the map. If the key does exist, the function will
-- insert @f new_value old_value@.
--
-- > insertWith (++) 5 "xxx" (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "xxxa")]
-- > insertWith (++) 7 "xxx" (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a"), (7, "xxx")]
-- > insertWith (++) 5 "xxx" empty                         == singleton 5 "xxx"

insertWith :: (a -> a -> a) -> Key -> a -> IntMap a -> IntMap a
insertWith f k x t
  = insertWithKey (\_ x' y' -> f x' y') k x t

-- | \(O(\min(n,W))\). Insert with a combining function.
-- @'insertWithKey' f key value mp@
-- will insert the pair (key, value) into @mp@ if key does
-- not exist in the map. If the key does exist, the function will
-- insert @f key new_value old_value@.
--
-- > let f key new_value old_value = (show key) ++ ":" ++ new_value ++ "|" ++ old_value
-- > insertWithKey f 5 "xxx" (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "5:xxx|a")]
-- > insertWithKey f 7 "xxx" (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a"), (7, "xxx")]
-- > insertWithKey f 5 "xxx" empty                         == singleton 5 "xxx"

insertWithKey :: (Key -> a -> a -> a) -> Key -> a -> IntMap a -> IntMap a
insertWithKey f !k x (NonEmpty m) = NonEmpty (NE.insertWithKey f k x m)
insertWithKey _ k x Nil = NonEmpty (NE.Tip k x)

-- | \(O(\min(n,W))\). The expression (@'insertLookupWithKey' f k x map@)
-- is a pair where the first element is equal to (@'lookup' k map@)
-- and the second element equal to (@'insertWithKey' f k x map@).
--
-- > let f key new_value old_value = (show key) ++ ":" ++ new_value ++ "|" ++ old_value
-- > insertLookupWithKey f 5 "xxx" (fromList [(5,"a"), (3,"b")]) == (Just "a", fromList [(3, "b"), (5, "5:xxx|a")])
-- > insertLookupWithKey f 7 "xxx" (fromList [(5,"a"), (3,"b")]) == (Nothing,  fromList [(3, "b"), (5, "a"), (7, "xxx")])
-- > insertLookupWithKey f 5 "xxx" empty                         == (Nothing,  singleton 5 "xxx")
--
-- This is how to define @insertLookup@ using @insertLookupWithKey@:
--
-- > let insertLookup kx x t = insertLookupWithKey (\_ a _ -> a) kx x t
-- > insertLookup 5 "x" (fromList [(5,"a"), (3,"b")]) == (Just "a", fromList [(3, "b"), (5, "x")])
-- > insertLookup 7 "x" (fromList [(5,"a"), (3,"b")]) == (Nothing,  fromList [(3, "b"), (5, "a"), (7, "x")])

insertLookupWithKey :: (Key -> a -> a -> a) -> Key -> a -> IntMap a -> (Maybe a, IntMap a)
insertLookupWithKey f !k x (NonEmpty m) = fmap NonEmpty (NE.insertLookupWithKey f k x m)
insertLookupWithKey _ k x Nil = (Nothing,NonEmpty (NE.Tip k x))


{--------------------------------------------------------------------
  Deletion
--------------------------------------------------------------------}
-- | \(O(\min(n,W))\). Delete a key and its value from the map. When the key is not
-- a member of the map, the original map is returned.
--
-- > delete 5 (fromList [(5,"a"), (3,"b")]) == singleton 3 "b"
-- > delete 7 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a")]
-- > delete 5 empty                         == empty

delete :: Key -> IntMap a -> IntMap a
delete !k t@(NonEmpty (NE.Bin p m l r))
  | nomatch k p m = t
  | zero k m      = binCheckLeft p m (delete k (NonEmpty l)) r
  | otherwise     = binCheckRight p m l (delete k (NonEmpty r))
delete k t@(NonEmpty (NE.Tip ky _))
  | k == ky       = Nil
  | otherwise     = t
delete _k Nil = Nil

-- | \(O(\min(n,W))\). Adjust a value at a specific key. When the key is not
-- a member of the map, the original map is returned.
--
-- > adjust ("new " ++) 5 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "new a")]
-- > adjust ("new " ++) 7 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a")]
-- > adjust ("new " ++) 7 empty                         == empty

adjust ::  (a -> a) -> Key -> IntMap a -> IntMap a
adjust f k m
  = adjustWithKey (\_ x -> f x) k m

-- | \(O(\min(n,W))\). Adjust a value at a specific key. When the key is not
-- a member of the map, the original map is returned.
--
-- > let f key x = (show key) ++ ":new " ++ x
-- > adjustWithKey f 5 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "5:new a")]
-- > adjustWithKey f 7 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a")]
-- > adjustWithKey f 7 empty                         == empty

adjustWithKey ::  (Key -> a -> a) -> Key -> IntMap a -> IntMap a
adjustWithKey f !k (NonEmpty m) = NonEmpty $ NE.adjustWithKey f k m
adjustWithKey _ _ Nil = Nil


-- | \(O(\min(n,W))\). The expression (@'update' f k map@) updates the value @x@
-- at @k@ (if it is in the map). If (@f x@) is 'Nothing', the element is
-- deleted. If it is (@'Just' y@), the key @k@ is bound to the new value @y@.
--
-- > let f x = if x == "a" then Just "new a" else Nothing
-- > update f 5 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "new a")]
-- > update f 7 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a")]
-- > update f 3 (fromList [(5,"a"), (3,"b")]) == singleton 5 "a"

update ::  (a -> Maybe a) -> Key -> IntMap a -> IntMap a
update f
  = updateWithKey (\_ x -> f x)

-- | \(O(\min(n,W))\). The expression (@'update' f k map@) updates the value @x@
-- at @k@ (if it is in the map). If (@f k x@) is 'Nothing', the element is
-- deleted. If it is (@'Just' y@), the key @k@ is bound to the new value @y@.
--
-- > let f k x = if x == "a" then Just ((show k) ++ ":new a") else Nothing
-- > updateWithKey f 5 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "5:new a")]
-- > updateWithKey f 7 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a")]
-- > updateWithKey f 3 (fromList [(5,"a"), (3,"b")]) == singleton 5 "a"

updateWithKey ::  (Key -> a -> Maybe a) -> Key -> IntMap a -> IntMap a
updateWithKey f !k (NonEmpty ne) = go ne
  where
    go (NE.Bin p m l r)
      | zero k m      = binCheckLeft p m (go l) r
      | otherwise     = binCheckRight p m l (go r)
    go t@(NE.Tip ky y)
      | k == ky       = case (f k y) of
                          Just y' -> NonEmpty (NE.Tip ky y')
                          Nothing -> Nil
      | otherwise     = NonEmpty t
updateWithKey _ _ Nil = Nil

-- | \(O(\min(n,W))\). Lookup and update.
-- The function returns original value, if it is updated.
-- This is different behavior than 'Data.Map.updateLookupWithKey'.
-- Returns the original key value if the map entry is deleted.
--
-- > let f k x = if x == "a" then Just ((show k) ++ ":new a") else Nothing
-- > updateLookupWithKey f 5 (fromList [(5,"a"), (3,"b")]) == (Just "a", fromList [(3, "b"), (5, "5:new a")])
-- > updateLookupWithKey f 7 (fromList [(5,"a"), (3,"b")]) == (Nothing,  fromList [(3, "b"), (5, "a")])
-- > updateLookupWithKey f 3 (fromList [(5,"a"), (3,"b")]) == (Just "b", singleton 5 "a")

updateLookupWithKey :: (Key -> a -> Maybe a) -> Key -> IntMap a -> (Maybe a,IntMap a)
updateLookupWithKey f !k (NonEmpty ne) = go ne
  where
    go (NE.Bin p m l r)
      | zero k m      = let !(found,l') = go l
                        in (found, binCheckLeft p m l' r)
      | otherwise     = let !(found,r') = go r
                        in (found, binCheckRight p m l r')
    go t@(NE.Tip ky y)
      | k==ky         = case f k y of
                          Just y' -> (Just y,NonEmpty (NE.Tip ky y'))
                          Nothing -> (Just y,Nil)
      | otherwise     = (Nothing, NonEmpty t)
updateLookupWithKey _ _ Nil = (Nothing,Nil)



-- | \(O(\min(n,W))\). The expression (@'alter' f k map@) alters the value @x@ at @k@, or absence thereof.
-- 'alter' can be used to insert, delete, or update a value in an 'IntMap'.
-- In short : @'lookup' k ('alter' f k m) = f ('lookup' k m)@.
alter :: forall a. (Maybe a -> Maybe a) -> Key -> IntMap a -> IntMap a
alter f !k (NonEmpty t) = go t
  where
    go ne@(NE.Bin p m l r)
      | nomatch k p m = case f Nothing of
                          Nothing -> NonEmpty t
                          Just x -> NonEmpty $ NE.link k (NE.Tip k x) p ne
      | zero k m      = binCheckLeft p m (go l) r
      | otherwise     = binCheckRight p m l (go r)
    go ne@(NE.Tip ky y)
      | k==ky         = case f (Just y) of
                          Just x -> NonEmpty (NE.Tip ky x)
                          Nothing -> Nil
      | otherwise     = case f Nothing of
                          Just x -> NonEmpty $ NE.link k (NE.Tip k x) ky ne
                          Nothing -> NonEmpty (NE.Tip ky y)
alter f k Nil     = case f Nothing of
                      Just x -> NonEmpty (NE.Tip k x)
                      Nothing -> Nil

-- | \(O(\min(n,W))\). The expression (@'alterF' f k map@) alters the value @x@ at
-- @k@, or absence thereof.  'alterF' can be used to inspect, insert, delete,
-- or update a value in an 'IntMap'.  In short : @'lookup' k <$> 'alterF' f k m = f
-- ('lookup' k m)@.
--
-- Example:
--
-- @
-- interactiveAlter :: Int -> IntMap String -> IO (IntMap String)
-- interactiveAlter k m = alterF f k m where
--   f Nothing = do
--      putStrLn $ show k ++
--          " was not found in the map. Would you like to add it?"
--      getUserResponse1 :: IO (Maybe String)
--   f (Just old) = do
--      putStrLn $ "The key is currently bound to " ++ show old ++
--          ". Would you like to change or delete it?"
--      getUserResponse2 :: IO (Maybe String)
-- @
--
-- 'alterF' is the most general operation for working with an individual
-- key that may or may not be in a given map.
--
-- Note: 'alterF' is a flipped version of the @at@ combinator from
-- @Control.Lens.At@.
--
-- @since 0.5.8

alterF :: Functor f
       => (Maybe a -> f (Maybe a)) -> Key -> IntMap a -> f (IntMap a)
-- This implementation was stolen from 'Control.Lens.At'.
alterF f k m = (<$> f mv) $ \fres ->
  case fres of
    Nothing -> maybe m (const (delete k m)) mv
    Just v' -> insert k v' m
  where mv = lookup k m

{--------------------------------------------------------------------
  Union
--------------------------------------------------------------------}
-- | The union of a list of maps.
--
-- > unions [(fromList [(5, "a"), (3, "b")]), (fromList [(5, "A"), (7, "C")]), (fromList [(5, "A3"), (3, "B3")])]
-- >     == fromList [(3, "b"), (5, "a"), (7, "C")]
-- > unions [(fromList [(5, "A3"), (3, "B3")]), (fromList [(5, "A"), (7, "C")]), (fromList [(5, "a"), (3, "b")])]
-- >     == fromList [(3, "B3"), (5, "A3"), (7, "C")]

unions :: Foldable f => f (IntMap a) -> IntMap a
unions xs
  = Foldable.foldl' union empty xs

-- | The union of a list of maps, with a combining operation.
--
-- > unionsWith (++) [(fromList [(5, "a"), (3, "b")]), (fromList [(5, "A"), (7, "C")]), (fromList [(5, "A3"), (3, "B3")])]
-- >     == fromList [(3, "bB3"), (5, "aAA3"), (7, "C")]

unionsWith :: Foldable f => (a->a->a) -> f (IntMap a) -> IntMap a
unionsWith f ts
  = Foldable.foldl' (unionWith f) empty ts

-- | \(O(n+m)\). The (left-biased) union of two maps.
-- It prefers the first map when duplicate keys are encountered,
-- i.e. (@'union' == 'unionWith' 'const'@).
--
-- > union (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == fromList [(3, "b"), (5, "a"), (7, "C")]

union :: IntMap a -> IntMap a -> IntMap a
union (NonEmpty m1) (NonEmpty m2) = NonEmpty $ NE.union m1 m2
union Nil m2 = m2
union m1 Nil = m1

-- | \(O(n+m)\). The union with a combining function.
--
-- > unionWith (++) (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == fromList [(3, "b"), (5, "aA"), (7, "C")]

unionWith :: (a -> a -> a) -> IntMap a -> IntMap a -> IntMap a
unionWith f m1 m2
  = unionWithKey (\_ x y -> f x y) m1 m2

-- | \(O(n+m)\). The union with a combining function.
--
-- > let f key left_value right_value = (show key) ++ ":" ++ left_value ++ "|" ++ right_value
-- > unionWithKey f (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == fromList [(3, "b"), (5, "5:a|A"), (7, "C")]

unionWithKey :: (Key -> a -> a -> a) -> IntMap a -> IntMap a -> IntMap a
unionWithKey f (NonEmpty m1) (NonEmpty m2)
  = NonEmpty $
      NE.mergeWithKey
        (\(NE.Tip k1 x1) (NE.Tip _k2 x2) -> NE.Tip k1 (f k1 x1 x2))
        id id m1 m2
unionWithKey _ Nil m2 = m2
unionWithKey _ m1 Nil = m1

{--------------------------------------------------------------------
  Difference
--------------------------------------------------------------------}
-- | \(O(n+m)\). Difference between two maps (based on keys).
--
-- > difference (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == singleton 3 "b"

difference :: IntMap a -> IntMap b -> IntMap a
difference m1 m2
  = mergeWithKey (\_ _ _ -> Nothing) NonEmpty (const Nil) m1 m2

-- | \(O(n+m)\). Difference with a combining function.
--
-- > let f al ar = if al == "b" then Just (al ++ ":" ++ ar) else Nothing
-- > differenceWith f (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (3, "B"), (7, "C")])
-- >     == singleton 3 "b:B"

differenceWith :: (a -> b -> Maybe a) -> IntMap a -> IntMap b -> IntMap a
differenceWith f m1 m2
  = differenceWithKey (\_ x y -> f x y) m1 m2

-- | \(O(n+m)\). Difference with a combining function. When two equal keys are
-- encountered, the combining function is applied to the key and both values.
-- If it returns 'Nothing', the element is discarded (proper set difference).
-- If it returns (@'Just' y@), the element is updated with a new value @y@.
--
-- > let f k al ar = if al == "b" then Just ((show k) ++ ":" ++ al ++ "|" ++ ar) else Nothing
-- > differenceWithKey f (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (3, "B"), (10, "C")])
-- >     == singleton 3 "3:b|B"

differenceWithKey :: (Key -> a -> b -> Maybe a) -> IntMap a -> IntMap b -> IntMap a
differenceWithKey f m1 m2
  = mergeWithKey f NonEmpty (const Nil) m1 m2


-- TODO(wrengr): re-verify that asymptotic bound
-- | \(O(n+m)\). Remove all the keys in a given set from a map.
--
-- @
-- m \`withoutKeys\` s = 'filterWithKey' (\\k _ -> k ``IntSet.notMember`` s) m
-- @
--
-- @since 0.5.8
withoutKeys :: IntMap a -> IntSet.IntSet -> IntMap a
withoutKeys (NonEmpty ne) ks = go ne ks
  where
  go t1@(NE.Bin p1 m1 l1 r1) t2@(IntSet.Bin p2 m2 l2 r2)
    | shorter m1 m2  = difference1
    | shorter m2 m1  = difference2
    | p1 == p2       = bin p1 m1 (go l1 l2) (go r1 r2)
    | otherwise      = NonEmpty t1
    where
    difference1
        | nomatch p2 p1 m1  = NonEmpty t1
        | zero p2 m1        = binCheckLeft p1 m1 (go l1 t2) r1
        | otherwise         = binCheckRight p1 m1 l1 (go r1 t2)
    difference2
        | nomatch p1 p2 m2  = NonEmpty t1
        | zero p1 m2        = go t1 l2
        | otherwise         = go t1 r2
  go t1@(NE.Bin p1 m1 _ _) (IntSet.Tip p2 bm2) =
    let minbit = bitmapOf p1
        lt_minbit = minbit - 1
        maxbit = bitmapOf (p1 .|. (m1 .|. (m1 - 1)))
        gt_maxbit = (-maxbit) `xor` maxbit
    -- TODO(wrengr): should we manually inline/unroll 'updatePrefix'
    -- and 'withoutBM' here, in order to avoid redundant case analyses?
    in updatePrefix p2 t1 $ withoutBM (bm2 .|. lt_minbit .|. gt_maxbit)
  go t1@(NE.Bin _ _ _ _) IntSet.Nil = NonEmpty t1
  go t1@(NE.Tip k1 _) t2
      | k1 `IntSet.member` t2 = Nil
      | otherwise = NonEmpty t1
withoutKeys Nil _ = Nil

updatePrefix
    :: IntSetPrefix -> NE.IntMap a -> (NE.IntMap a -> IntMap a) -> IntMap a
updatePrefix !kp t@(NE.Bin p m l r) f
    | m .&. IntSet.suffixBitMask /= 0 =
        if p .&. IntSet.prefixBitMask == kp then f t else NonEmpty t
    | nomatch kp p m = NonEmpty t
    | zero kp m      = binCheckLeft p m (updatePrefix kp l f) r
    | otherwise      = binCheckRight p m l (updatePrefix kp r f)
updatePrefix kp t@(NE.Tip kx _) f
    | kx .&. IntSet.prefixBitMask == kp = f t
    | otherwise = NonEmpty t

withoutBM :: IntSetBitMap -> NE.IntMap a -> IntMap a
withoutBM 0 t = NonEmpty t
withoutBM bm (NE.Bin p m l r) =
    let leftBits = bitmapOf (p .|. m) - 1
        bmL = bm .&. leftBits
        bmR = bm `xor` bmL -- = (bm .&. complement leftBits)
   in  bin p m (withoutBM bmL l) (withoutBM bmR r)
withoutBM bm t@(NE.Tip k _)
    -- TODO(wrengr): need we manually inline 'IntSet.Member' here?
    | k `IntSet.member` IntSet.Tip (k .&. IntSet.prefixBitMask) bm = Nil
    | otherwise = NonEmpty t

{--------------------------------------------------------------------
  Intersection
--------------------------------------------------------------------}
-- | \(O(n+m)\). The (left-biased) intersection of two maps (based on keys).
--
-- > intersection (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == singleton 5 "a"

intersection :: IntMap a -> IntMap b -> IntMap a
intersection m1 m2
  = mergeWithKey' bin (\ a _ -> NonEmpty a) (const Nil) (const Nil) m1 m2


-- TODO(wrengr): re-verify that asymptotic bound
-- | \(O(n+m)\). The restriction of a map to the keys in a set.
--
-- @
-- m \`restrictKeys\` s = 'filterWithKey' (\\k _ -> k ``IntSet.member`` s) m
-- @
--
-- @since 0.5.8
restrictKeys :: IntMap a -> IntSet.IntSet -> IntMap a
restrictKeys (NonEmpty t0) ks = go t0 ks
  where
    go t1@(NE.Bin p1 m1 l1 r1) t2@(IntSet.Bin p2 m2 l2 r2)
      | shorter m1 m2  = intersection1
      | shorter m2 m1  = intersection2
      | p1 == p2       = bin p1 m1 (go l1 l2) (go r1 r2)
      | otherwise      = Nil
      where
      intersection1
          | nomatch p2 p1 m1  = Nil
          | zero p2 m1        = go l1 t2
          | otherwise         = go r1 t2
      intersection2
          | nomatch p1 p2 m2  = Nil
          | zero p1 m2        = go t1 l2
          | otherwise         = go t1 r2
    go t1@(NE.Bin p1 m1 _ _) (IntSet.Tip p2 bm2) =
      let minbit = bitmapOf p1
          ge_minbit = complement (minbit - 1)
          maxbit = bitmapOf (p1 .|. (m1 .|. (m1 - 1)))
          le_maxbit = maxbit .|. (maxbit - 1)
      -- TODO(wrengr): should we manually inline/unroll 'lookupPrefix'
      -- and 'restrictBM' here, in order to avoid redundant case analyses?
      in restrictBM (bm2 .&. ge_minbit .&. le_maxbit) (lookupPrefix p2 $ NonEmpty t1)
    go (NE.Bin _ _ _ _) IntSet.Nil = Nil
    go t1@(NE.Tip k1 _) t2
        | k1 `IntSet.member` t2 = NonEmpty t1
        | otherwise = Nil
restrictKeys Nil _ = Nil

-- | \(O(\min(n,W))\). Restrict to the sub-map with all keys matching
-- a key prefix.
lookupPrefix :: IntSetPrefix -> IntMap a -> IntMap a
lookupPrefix !kp (NonEmpty t0) = go t0
  where
    go t@(NE.Bin p m l r)
      | m .&. IntSet.suffixBitMask /= 0 =
          if p .&. IntSet.prefixBitMask == kp then NonEmpty t else Nil
      | nomatch kp p m = Nil
      | zero kp m      = go l
      | otherwise      = go r
    go t@(NE.Tip kx _)
      | (kx .&. IntSet.prefixBitMask) == kp = NonEmpty t
      | otherwise = Nil
lookupPrefix _ Nil = Nil

restrictBM :: IntSetBitMap -> IntMap a -> IntMap a
restrictBM 0 _ = Nil
restrictBM bm0 (NonEmpty t0) = go bm0 t0
  where
    go bm (NE.Bin p m l r) =
      let leftBits = bitmapOf (p .|. m) - 1
          bmL = bm .&. leftBits
          bmR = bm `xor` bmL -- = (bm .&. complement leftBits)
      in  bin p m (go bmL l) (go bmR r)
    go bm t@(NE.Tip k _)
    -- TODO(wrengr): need we manually inline 'IntSet.Member' here?
      | k `IntSet.member` IntSet.Tip (k .&. IntSet.prefixBitMask) bm = NonEmpty t
      | otherwise = Nil
restrictBM _ Nil = Nil

-- | \(O(n+m)\). The intersection with a combining function.
--
-- > intersectionWith (++) (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == singleton 5 "aA"

intersectionWith :: (a -> b -> c) -> IntMap a -> IntMap b -> IntMap c
intersectionWith f m1 m2
  = intersectionWithKey (\_ x y -> f x y) m1 m2

-- | \(O(n+m)\). The intersection with a combining function.
--
-- > let f k al ar = (show k) ++ ":" ++ al ++ "|" ++ ar
-- > intersectionWithKey f (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == singleton 5 "5:a|A"

intersectionWithKey :: (Key -> a -> b -> c) -> IntMap a -> IntMap b -> IntMap c
intersectionWithKey f m1 m2
  = mergeWithKey' bin
    (\(NE.Tip k1 x1) (NE.Tip _k2 x2) -> NonEmpty $ NE.Tip k1 (f k1 x1 x2))
    (const Nil) (const Nil) m1 m2

{--------------------------------------------------------------------
  MergeWithKey
--------------------------------------------------------------------}

-- | \(O(n+m)\). A high-performance universal combining function. Using
-- 'mergeWithKey', all combining functions can be defined without any loss of
-- efficiency (with exception of 'union', 'difference' and 'intersection',
-- where sharing of some nodes is lost with 'mergeWithKey').
--
-- Please make sure you know what is going on when using 'mergeWithKey',
-- otherwise you can be surprised by unexpected code growth or even
-- corruption of the data structure.
--
-- When 'mergeWithKey' is given three arguments, it is inlined to the call
-- site. You should therefore use 'mergeWithKey' only to define your custom
-- combining functions. For example, you could define 'unionWithKey',
-- 'differenceWithKey' and 'intersectionWithKey' as
--
-- > myUnionWithKey f m1 m2 = mergeWithKey (\k x1 x2 -> Just (f k x1 x2)) id id m1 m2
-- > myDifferenceWithKey f m1 m2 = mergeWithKey f id (const empty) m1 m2
-- > myIntersectionWithKey f m1 m2 = mergeWithKey (\k x1 x2 -> Just (f k x1 x2)) (const empty) (const empty) m1 m2
--
-- When calling @'mergeWithKey' combine only1 only2@, a function combining two
-- 'IntMap's is created, such that
--
-- * if a key is present in both maps, it is passed with both corresponding
--   values to the @combine@ function. Depending on the result, the key is either
--   present in the result with specified value, or is left out;
--
-- * a nonempty subtree present only in the first map is passed to @only1@ and
--   the output is added to the result;
--
-- * a nonempty subtree present only in the second map is passed to @only2@ and
--   the output is added to the result.
--
-- The @only1@ and @only2@ methods /must return a map with a subset (possibly empty) of the keys of the given map/.
-- The values can be modified arbitrarily. Most common variants of @only1@ and
-- @only2@ are 'id' and @'const' 'empty'@, but for example @'map' f@ or
-- @'filterWithKey' f@ could be used for any @f@.

mergeWithKey :: (Key -> a -> b -> Maybe c) -> (NE.IntMap a -> IntMap c) -> (NE.IntMap b -> IntMap c)
             -> IntMap a -> IntMap b -> IntMap c
mergeWithKey f g1 g2 = mergeWithKey' bin combine g1 g2
  where -- We use the lambda form to avoid non-exhaustive pattern matches warning.
        combine = \(NE.Tip k1 x1) (NE.Tip _k2 x2) ->
          case f k1 x1 x2 of
            Nothing -> Nil
            Just x  -> NonEmpty $ NE.Tip k1 x
        {-# INLINE combine #-}
{-# INLINE mergeWithKey #-}

-- Slightly more general version of mergeWithKey. It differs in the following:
--
-- * the combining function operates on maps instead of keys and values. The
--   reason is to enable sharing in union, difference and intersection.
--
-- * mergeWithKey' is given an equivalent of bin. The reason is that in union*,
--   Bin constructor can be used, because we know both subtrees are nonempty.

mergeWithKey' :: forall a b c. (Prefix -> Mask -> IntMap c -> IntMap c -> IntMap c)
              -> (NE.IntMap a -> NE.IntMap b -> IntMap c)
              -> (NE.IntMap a -> IntMap c) -> (NE.IntMap b -> IntMap c)
              -> IntMap a -> IntMap b -> IntMap c
mergeWithKey' bin' f g1 g2 = go
  where
    go :: IntMap a -> IntMap b -> IntMap c
    go (NonEmpty t1) (NonEmpty t2) = go' t1 t2
    go (NonEmpty t1) Nil = g1 t1
    go Nil (NonEmpty t2) = g2 t2
    go Nil Nil = Nil

    go' :: NE.IntMap a -> NE.IntMap b -> IntMap c
    go' t1@(NE.Bin p1 m1 l1 r1) t2@(NE.Bin p2 m2 l2 r2)
      | shorter m1 m2  = merge1
      | shorter m2 m1  = merge2
      | p1 == p2       = bin' p1 m1 (go' l1 l2) (go' r1 r2)
      | otherwise      = maybe_link p1 (g1 t1) p2 (g2 t2)
      where
        merge1 | nomatch p2 p1 m1  = maybe_link p1 (g1 t1) p2 (g2 t2)
               | zero p2 m1        = bin' p1 m1 (go' l1 t2) (g1 r1)
               | otherwise         = bin' p1 m1 (g1 l1) (go' r1 t2)
        merge2 | nomatch p1 p2 m2  = maybe_link p1 (g1 t1) p2 (g2 t2)
               | zero p1 m2        = bin' p2 m2 (go' t1 l2) (g2 r2)
               | otherwise         = bin' p2 m2 (g2 l2) (go' t1 r2)

    go' t1'@(NE.Bin _ _ _ _) t2'@(NE.Tip k2' _) = merge0 t2' k2' t1'
      where
        merge0 t2 k2 t1@(NE.Bin p1 m1 l1 r1)
          | nomatch k2 p1 m1 = maybe_link p1 (g1 t1) k2 (g2 t2)
          | zero k2 m1 = bin' p1 m1 (merge0 t2 k2 l1) (g1 r1)
          | otherwise  = bin' p1 m1 (g1 l1) (merge0 t2 k2 r1)
        merge0 t2 k2 t1@(NE.Tip k1 _)
          | k1 == k2 = f t1 t2
          | otherwise = maybe_link k1 (g1 t1) k2 (g2 t2)

    go' t1'@(NE.Tip k1' _) t2' = merge0 t1' k1' t2'
      where
        merge0 t1 k1 t2@(NE.Bin p2 m2 l2 r2)
          | nomatch k1 p2 m2 = maybe_link k1 (g1 t1) p2 (g2 t2)
          | zero k1 m2 = bin' p2 m2 (merge0 t1 k1 l2) (g2 r2)
          | otherwise  = bin' p2 m2 (g2 l2) (merge0 t1 k1 r2)
        merge0 t1 k1 t2@(NE.Tip k2 _)
          | k1 == k2 = f t1 t2
          | otherwise = maybe_link k1 (g1 t1) k2 (g2 t2)

    maybe_link _ Nil _ t2 = t2
    maybe_link _ t1 _ Nil = t1
    maybe_link p1 (NonEmpty t1) p2 (NonEmpty t2) =
      NonEmpty $ NE.link p1 t1 p2 t2
    {-# INLINE maybe_link #-}
{-# INLINE mergeWithKey' #-}


{--------------------------------------------------------------------
  mergeA
--------------------------------------------------------------------}

-- | A tactic for dealing with keys present in one map but not the
-- other in 'merge' or 'mergeA'.
--
-- A tactic of type @WhenMissing f k x z@ is an abstract representation
-- of a function of type @Key -> x -> f (Maybe z)@.
--
-- @since 0.5.9

data WhenMissing f x y = WhenMissing
  { missingSubtree :: NE.IntMap x -> f (IntMap y)
  , missingKey :: Key -> x -> f (Maybe y)}

-- | @since 0.5.9
instance (Applicative f, Monad f) => Functor (WhenMissing f x) where
  fmap = mapWhenMissing
  {-# INLINE fmap #-}


-- | @since 0.5.9
instance (Applicative f, Monad f) => Category.Category (WhenMissing f)
  where
    id = preserveMissing
    f . g =
      traverseMaybeMissing $ \ k x -> do
        y <- missingKey g k x
        case y of
          Nothing -> pure Nothing
          Just q  -> missingKey f k q
    {-# INLINE id #-}
    {-# INLINE (.) #-}


-- | Equivalent to @ReaderT k (ReaderT x (MaybeT f))@.
--
-- @since 0.5.9
instance (Applicative f, Monad f) => Applicative (WhenMissing f x) where
  pure x = mapMissing (\ _ _ -> x)
  f <*> g =
    traverseMaybeMissing $ \k x -> do
      res1 <- missingKey f k x
      case res1 of
        Nothing -> pure Nothing
        Just r  -> (pure $!) . fmap r =<< missingKey g k x
  {-# INLINE pure #-}
  {-# INLINE (<*>) #-}


-- | Equivalent to @ReaderT k (ReaderT x (MaybeT f))@.
--
-- @since 0.5.9
instance (Applicative f, Monad f) => Monad (WhenMissing f x) where
  m >>= f =
    traverseMaybeMissing $ \k x -> do
      res1 <- missingKey m k x
      case res1 of
        Nothing -> pure Nothing
        Just r  -> missingKey (f r) k x
  {-# INLINE (>>=) #-}


-- | Map covariantly over a @'WhenMissing' f x@.
--
-- @since 0.5.9
mapWhenMissing
  :: (Applicative f, Monad f)
  => (a -> b)
  -> WhenMissing f x a
  -> WhenMissing f x b
mapWhenMissing f t = WhenMissing
  { missingSubtree = \m -> missingSubtree t m >>= \m' -> pure $! fmap f m'
  , missingKey     = \k x -> missingKey t k x >>= \q -> (pure $! fmap f q) }
{-# INLINE mapWhenMissing #-}


-- | Map covariantly over a @'WhenMissing' f x@, using only a
-- 'Functor f' constraint.
mapGentlyWhenMissing
  :: Functor f
  => (a -> b)
  -> WhenMissing f x a
  -> WhenMissing f x b
mapGentlyWhenMissing f t = WhenMissing
  { missingSubtree = \m -> fmap f <$> missingSubtree t m
  , missingKey     = \k x -> fmap f <$> missingKey t k x }
{-# INLINE mapGentlyWhenMissing #-}


-- | Map covariantly over a @'WhenMatched' f k x@, using only a
-- 'Functor f' constraint.
mapGentlyWhenMatched
  :: Functor f
  => (a -> b)
  -> WhenMatched f x y a
  -> WhenMatched f x y b
mapGentlyWhenMatched f t =
  zipWithMaybeAMatched $ \k x y -> fmap f <$> runWhenMatched t k x y
{-# INLINE mapGentlyWhenMatched #-}


-- | Map contravariantly over a @'WhenMissing' f _ x@.
--
-- @since 0.5.9
lmapWhenMissing :: (b -> a) -> WhenMissing f a x -> WhenMissing f b x
lmapWhenMissing f t = WhenMissing
  { missingSubtree = \m -> missingSubtree t (fmap f m)
  , missingKey     = \k x -> missingKey t k (f x) }
{-# INLINE lmapWhenMissing #-}


-- | Map contravariantly over a @'WhenMatched' f _ y z@.
--
-- @since 0.5.9
contramapFirstWhenMatched
  :: (b -> a)
  -> WhenMatched f a y z
  -> WhenMatched f b y z
contramapFirstWhenMatched f t =
  WhenMatched $ \k x y -> runWhenMatched t k (f x) y
{-# INLINE contramapFirstWhenMatched #-}


-- | Map contravariantly over a @'WhenMatched' f x _ z@.
--
-- @since 0.5.9
contramapSecondWhenMatched
  :: (b -> a)
  -> WhenMatched f x a z
  -> WhenMatched f x b z
contramapSecondWhenMatched f t =
  WhenMatched $ \k x y -> runWhenMatched t k x (f y)
{-# INLINE contramapSecondWhenMatched #-}


-- | A tactic for dealing with keys present in one map but not the
-- other in 'merge'.
--
-- A tactic of type @SimpleWhenMissing x z@ is an abstract
-- representation of a function of type @Key -> x -> Maybe z@.
--
-- @since 0.5.9
type SimpleWhenMissing = WhenMissing Identity


-- | A tactic for dealing with keys present in both maps in 'merge'
-- or 'mergeA'.
--
-- A tactic of type @WhenMatched f x y z@ is an abstract representation
-- of a function of type @Key -> x -> y -> f (Maybe z)@.
--
-- @since 0.5.9
newtype WhenMatched f x y z = WhenMatched
  { matchedKey :: Key -> x -> y -> f (Maybe z) }


-- | Along with zipWithMaybeAMatched, witnesses the isomorphism
-- between @WhenMatched f x y z@ and @Key -> x -> y -> f (Maybe z)@.
--
-- @since 0.5.9
runWhenMatched :: WhenMatched f x y z -> Key -> x -> y -> f (Maybe z)
runWhenMatched = matchedKey
{-# INLINE runWhenMatched #-}


-- | Along with traverseMaybeMissing, witnesses the isomorphism
-- between @WhenMissing f x y@ and @Key -> x -> f (Maybe y)@.
--
-- @since 0.5.9
runWhenMissing :: WhenMissing f x y -> Key-> x -> f (Maybe y)
runWhenMissing = missingKey
{-# INLINE runWhenMissing #-}


-- | @since 0.5.9
instance Functor f => Functor (WhenMatched f x y) where
  fmap = mapWhenMatched
  {-# INLINE fmap #-}


-- | @since 0.5.9
instance (Monad f, Applicative f) => Category.Category (WhenMatched f x)
  where
    id = zipWithMatched (\_ _ y -> y)
    f . g =
      zipWithMaybeAMatched $ \k x y -> do
        res <- runWhenMatched g k x y
        case res of
          Nothing -> pure Nothing
          Just r  -> runWhenMatched f k x r
    {-# INLINE id #-}
    {-# INLINE (.) #-}


-- | Equivalent to @ReaderT Key (ReaderT x (ReaderT y (MaybeT f)))@
--
-- @since 0.5.9
instance (Monad f, Applicative f) => Applicative (WhenMatched f x y) where
  pure x = zipWithMatched (\_ _ _ -> x)
  fs <*> xs =
    zipWithMaybeAMatched $ \k x y -> do
      res <- runWhenMatched fs k x y
      case res of
        Nothing -> pure Nothing
        Just r  -> (pure $!) . fmap r =<< runWhenMatched xs k x y
  {-# INLINE pure #-}
  {-# INLINE (<*>) #-}


-- | Equivalent to @ReaderT Key (ReaderT x (ReaderT y (MaybeT f)))@
--
-- @since 0.5.9
instance (Monad f, Applicative f) => Monad (WhenMatched f x y) where
  m >>= f =
    zipWithMaybeAMatched $ \k x y -> do
      res <- runWhenMatched m k x y
      case res of
        Nothing -> pure Nothing
        Just r  -> runWhenMatched (f r) k x y
  {-# INLINE (>>=) #-}


-- | Map covariantly over a @'WhenMatched' f x y@.
--
-- @since 0.5.9
mapWhenMatched
  :: Functor f
  => (a -> b)
  -> WhenMatched f x y a
  -> WhenMatched f x y b
mapWhenMatched f (WhenMatched g) =
  WhenMatched $ \k x y -> fmap (fmap f) (g k x y)
{-# INLINE mapWhenMatched #-}


-- | A tactic for dealing with keys present in both maps in 'merge'.
--
-- A tactic of type @SimpleWhenMatched x y z@ is an abstract
-- representation of a function of type @Key -> x -> y -> Maybe z@.
--
-- @since 0.5.9
type SimpleWhenMatched = WhenMatched Identity


-- | When a key is found in both maps, apply a function to the key
-- and values and use the result in the merged map.
--
-- > zipWithMatched
-- >   :: (Key -> x -> y -> z)
-- >   -> SimpleWhenMatched x y z
--
-- @since 0.5.9
zipWithMatched
  :: Applicative f
  => (Key -> x -> y -> z)
  -> WhenMatched f x y z
zipWithMatched f = WhenMatched $ \ k x y -> pure . Just $ f k x y
{-# INLINE zipWithMatched #-}


-- | When a key is found in both maps, apply a function to the key
-- and values to produce an action and use its result in the merged
-- map.
--
-- @since 0.5.9
zipWithAMatched
  :: Functor f
  => (Key -> x -> y -> f z)
  -> WhenMatched f x y z
zipWithAMatched f = WhenMatched $ \ k x y -> Just <$> f k x y
{-# INLINE zipWithAMatched #-}


-- | When a key is found in both maps, apply a function to the key
-- and values and maybe use the result in the merged map.
--
-- > zipWithMaybeMatched
-- >   :: (Key -> x -> y -> Maybe z)
-- >   -> SimpleWhenMatched x y z
--
-- @since 0.5.9
zipWithMaybeMatched
  :: Applicative f
  => (Key -> x -> y -> Maybe z)
  -> WhenMatched f x y z
zipWithMaybeMatched f = WhenMatched $ \ k x y -> pure $ f k x y
{-# INLINE zipWithMaybeMatched #-}


-- | When a key is found in both maps, apply a function to the key
-- and values, perform the resulting action, and maybe use the
-- result in the merged map.
--
-- This is the fundamental 'WhenMatched' tactic.
--
-- @since 0.5.9
zipWithMaybeAMatched
  :: (Key -> x -> y -> f (Maybe z))
  -> WhenMatched f x y z
zipWithMaybeAMatched f = WhenMatched $ \ k x y -> f k x y
{-# INLINE zipWithMaybeAMatched #-}


-- | Drop all the entries whose keys are missing from the other
-- map.
--
-- > dropMissing :: SimpleWhenMissing x y
--
-- prop> dropMissing = mapMaybeMissing (\_ _ -> Nothing)
--
-- but @dropMissing@ is much faster.
--
-- @since 0.5.9
dropMissing :: Applicative f => WhenMissing f x y
dropMissing = WhenMissing
  { missingSubtree = const (pure Nil)
  , missingKey     = \_ _ -> pure Nothing }
{-# INLINE dropMissing #-}


-- | Preserve, unchanged, the entries whose keys are missing from
-- the other map.
--
-- > preserveMissing :: SimpleWhenMissing x x
--
-- prop> preserveMissing = Merge.Lazy.mapMaybeMissing (\_ x -> Just x)
--
-- but @preserveMissing@ is much faster.
--
-- @since 0.5.9
preserveMissing :: Applicative f => WhenMissing f x x
preserveMissing = WhenMissing
  { missingSubtree = pure . NonEmpty
  , missingKey     = \_ v -> pure (Just v) }
{-# INLINE preserveMissing #-}


-- | Map over the entries whose keys are missing from the other map.
--
-- > mapMissing :: (k -> x -> y) -> SimpleWhenMissing x y
--
-- prop> mapMissing f = mapMaybeMissing (\k x -> Just $ f k x)
--
-- but @mapMissing@ is somewhat faster.
--
-- @since 0.5.9
mapMissing :: Applicative f => (Key -> x -> y) -> WhenMissing f x y
mapMissing f = WhenMissing
  { missingSubtree = \m -> pure $! mapWithKey f (NonEmpty m)
  , missingKey     = \k x -> pure $ Just (f k x) }
{-# INLINE mapMissing #-}


-- | Map over the entries whose keys are missing from the other
-- map, optionally removing some. This is the most powerful
-- 'SimpleWhenMissing' tactic, but others are usually more efficient.
--
-- > mapMaybeMissing :: (Key -> x -> Maybe y) -> SimpleWhenMissing x y
--
-- prop> mapMaybeMissing f = traverseMaybeMissing (\k x -> pure (f k x))
--
-- but @mapMaybeMissing@ uses fewer unnecessary 'Applicative'
-- operations.
--
-- @since 0.5.9
mapMaybeMissing
  :: Applicative f => (Key -> x -> Maybe y) -> WhenMissing f x y
mapMaybeMissing f = WhenMissing
  { missingSubtree = \m -> pure $! mapMaybeWithKey f (NonEmpty m)
  , missingKey     = \k x -> pure $! f k x }
{-# INLINE mapMaybeMissing #-}


-- | Filter the entries whose keys are missing from the other map.
--
-- > filterMissing :: (k -> x -> Bool) -> SimpleWhenMissing x x
--
-- prop> filterMissing f = Merge.Lazy.mapMaybeMissing $ \k x -> guard (f k x) *> Just x
--
-- but this should be a little faster.
--
-- @since 0.5.9
filterMissing
  :: Applicative f => (Key -> x -> Bool) -> WhenMissing f x x
filterMissing f = WhenMissing
  { missingSubtree = \m -> pure $! filterWithKey f (NonEmpty m)
  , missingKey     = \k x -> pure $! if f k x then Just x else Nothing }
{-# INLINE filterMissing #-}


-- | Filter the entries whose keys are missing from the other map
-- using some 'Applicative' action.
--
-- > filterAMissing f = Merge.Lazy.traverseMaybeMissing $
-- >   \k x -> (\b -> guard b *> Just x) <$> f k x
--
-- but this should be a little faster.
--
-- @since 0.5.9
filterAMissing
  :: Applicative f => (Key -> x -> f Bool) -> WhenMissing f x x
filterAMissing f = WhenMissing
  { missingSubtree = \m -> filterWithKeyA f (NonEmpty m)
  , missingKey     = \k x -> bool Nothing (Just x) <$> f k x }
{-# INLINE filterAMissing #-}


-- | \(O(n)\). Filter keys and values using an 'Applicative' predicate.
filterWithKeyA
  :: Applicative f => (Key -> a -> f Bool) -> IntMap a -> f (IntMap a)
filterWithKeyA _ Nil           = pure Nil
filterWithKeyA f t@(NonEmpty (NE.Tip k x))   = (\b -> if b then t else Nil) <$> f k x
filterWithKeyA f (NonEmpty (NE.Bin p m l r))
  | m < 0     = liftA2 (flip (bin p m)) (filterWithKeyA f (NonEmpty r)) (filterWithKeyA f (NonEmpty l))
  | otherwise = liftA2 (bin p m) (filterWithKeyA f (NonEmpty l)) (filterWithKeyA f (NonEmpty r))

-- | This wasn't in Data.Bool until 4.7.0, so we define it here
bool :: a -> a -> Bool -> a
bool f _ False = f
bool _ t True  = t


-- | Traverse over the entries whose keys are missing from the other
-- map.
--
-- @since 0.5.9
traverseMissing
  :: Applicative f => (Key -> x -> f y) -> WhenMissing f x y
traverseMissing f = WhenMissing
  { missingSubtree = traverseWithKey f . NonEmpty
  , missingKey = \k x -> Just <$> f k x }
{-# INLINE traverseMissing #-}


-- | Traverse over the entries whose keys are missing from the other
-- map, optionally producing values to put in the result. This is
-- the most powerful 'WhenMissing' tactic, but others are usually
-- more efficient.
--
-- @since 0.5.9
traverseMaybeMissing
  :: Applicative f => (Key -> x -> f (Maybe y)) -> WhenMissing f x y
traverseMaybeMissing f = WhenMissing
  { missingSubtree = traverseMaybeWithKey f . NonEmpty
  , missingKey = f }
{-# INLINE traverseMaybeMissing #-}


-- | \(O(n)\). Traverse keys\/values and collect the 'Just' results.
--
-- @since 0.6.4
traverseMaybeWithKey
  :: Applicative f => (Key -> a -> f (Maybe b)) -> IntMap a -> f (IntMap b)
traverseMaybeWithKey f = go
  where
    go (NonEmpty m) = go' m
    go Nil = pure Nil
    go' (NE.Tip k x)
      = maybe Nil (NonEmpty . NE.Tip k) <$> f k x
    go' (NE.Bin p m l r)
      | m < 0     = liftA2 (flip (bin p m)) (go' r) (go' l)
      | otherwise = liftA2 (bin p m) (go' l) (go' r)


-- | Merge two maps.
--
-- 'merge' takes two 'WhenMissing' tactics, a 'WhenMatched' tactic
-- and two maps. It uses the tactics to merge the maps. Its behavior
-- is best understood via its fundamental tactics, 'mapMaybeMissing'
-- and 'zipWithMaybeMatched'.
--
-- Consider
--
-- @
-- merge (mapMaybeMissing g1)
--              (mapMaybeMissing g2)
--              (zipWithMaybeMatched f)
--              m1 m2
-- @
--
-- Take, for example,
--
-- @
-- m1 = [(0, \'a\'), (1, \'b\'), (3, \'c\'), (4, \'d\')]
-- m2 = [(1, "one"), (2, "two"), (4, "three")]
-- @
--
-- 'merge' will first \"align\" these maps by key:
--
-- @
-- m1 = [(0, \'a\'), (1, \'b\'),               (3, \'c\'), (4, \'d\')]
-- m2 =           [(1, "one"), (2, "two"),           (4, "three")]
-- @
--
-- It will then pass the individual entries and pairs of entries
-- to @g1@, @g2@, or @f@ as appropriate:
--
-- @
-- maybes = [g1 0 \'a\', f 1 \'b\' "one", g2 2 "two", g1 3 \'c\', f 4 \'d\' "three"]
-- @
--
-- This produces a 'Maybe' for each key:
--
-- @
-- keys =     0        1          2           3        4
-- results = [Nothing, Just True, Just False, Nothing, Just True]
-- @
--
-- Finally, the @Just@ results are collected into a map:
--
-- @
-- return value = [(1, True), (2, False), (4, True)]
-- @
--
-- The other tactics below are optimizations or simplifications of
-- 'mapMaybeMissing' for special cases. Most importantly,
--
-- * 'dropMissing' drops all the keys.
-- * 'preserveMissing' leaves all the entries alone.
--
-- When 'merge' is given three arguments, it is inlined at the call
-- site. To prevent excessive inlining, you should typically use
-- 'merge' to define your custom combining functions.
--
--
-- Examples:
--
-- prop> unionWithKey f = merge preserveMissing preserveMissing (zipWithMatched f)
-- prop> intersectionWithKey f = merge dropMissing dropMissing (zipWithMatched f)
-- prop> differenceWith f = merge diffPreserve diffDrop f
-- prop> symmetricDifference = merge diffPreserve diffPreserve (\ _ _ _ -> Nothing)
-- prop> mapEachPiece f g h = merge (diffMapWithKey f) (diffMapWithKey g)
--
-- @since 0.5.9
merge
  :: SimpleWhenMissing a c -- ^ What to do with keys in @m1@ but not @m2@
  -> SimpleWhenMissing b c -- ^ What to do with keys in @m2@ but not @m1@
  -> SimpleWhenMatched a b c -- ^ What to do with keys in both @m1@ and @m2@
  -> IntMap a -- ^ Map @m1@
  -> IntMap b -- ^ Map @m2@
  -> IntMap c
merge g1 g2 f m1 m2 =
  runIdentity $ mergeA g1 g2 f m1 m2
{-# INLINE merge #-}


-- | An applicative version of 'merge'.
--
-- 'mergeA' takes two 'WhenMissing' tactics, a 'WhenMatched'
-- tactic and two maps. It uses the tactics to merge the maps.
-- Its behavior is best understood via its fundamental tactics,
-- 'traverseMaybeMissing' and 'zipWithMaybeAMatched'.
--
-- Consider
--
-- @
-- mergeA (traverseMaybeMissing g1)
--               (traverseMaybeMissing g2)
--               (zipWithMaybeAMatched f)
--               m1 m2
-- @
--
-- Take, for example,
--
-- @
-- m1 = [(0, \'a\'), (1, \'b\'), (3,\'c\'), (4, \'d\')]
-- m2 = [(1, "one"), (2, "two"), (4, "three")]
-- @
--
-- 'mergeA' will first \"align\" these maps by key:
--
-- @
-- m1 = [(0, \'a\'), (1, \'b\'),               (3, \'c\'), (4, \'d\')]
-- m2 =           [(1, "one"), (2, "two"),           (4, "three")]
-- @
--
-- It will then pass the individual entries and pairs of entries
-- to @g1@, @g2@, or @f@ as appropriate:
--
-- @
-- actions = [g1 0 \'a\', f 1 \'b\' "one", g2 2 "two", g1 3 \'c\', f 4 \'d\' "three"]
-- @
--
-- Next, it will perform the actions in the @actions@ list in order from
-- left to right.
--
-- @
-- keys =     0        1          2           3        4
-- results = [Nothing, Just True, Just False, Nothing, Just True]
-- @
--
-- Finally, the @Just@ results are collected into a map:
--
-- @
-- return value = [(1, True), (2, False), (4, True)]
-- @
--
-- The other tactics below are optimizations or simplifications of
-- 'traverseMaybeMissing' for special cases. Most importantly,
--
-- * 'dropMissing' drops all the keys.
-- * 'preserveMissing' leaves all the entries alone.
-- * 'mapMaybeMissing' does not use the 'Applicative' context.
--
-- When 'mergeA' is given three arguments, it is inlined at the call
-- site. To prevent excessive inlining, you should generally only use
-- 'mergeA' to define custom combining functions.
--
-- @since 0.5.9
mergeA
  :: (Applicative f)
  => WhenMissing f a c -- ^ What to do with keys in @m1@ but not @m2@
  -> WhenMissing f b c -- ^ What to do with keys in @m2@ but not @m1@
  -> WhenMatched f a b c -- ^ What to do with keys in both @m1@ and @m2@
  -> IntMap a -- ^ Map @m1@
  -> IntMap b -- ^ Map @m2@
  -> f (IntMap c)
mergeA
    WhenMissing{missingSubtree = g1t, missingKey = g1k}
    WhenMissing{missingSubtree = g2t, missingKey = g2k}
    WhenMatched{matchedKey = f}
    = go
  where
    go (NonEmpty t1) (NonEmpty t2) = go' t1 t2
    go (NonEmpty t1) Nil = g1t t1
    go Nil (NonEmpty t2) = g2t t2
    go Nil Nil = pure Nil

    -- This case is already covered below.
    -- go' (NE.Tip k1 x1) (NE.Tip k2 x2) = mergeTips k1 x1 k2 x2

    go' (NE.Tip k1 x1) t2' = merge2 t2'
      where
        merge2 t2@(NE.Bin p2 m2 l2 r2)
          | nomatch k1 p2 m2 = linkA k1 (subsingletonBy g1k k1 x1) p2 (g2t t2)
          | zero k1 m2       = binA p2 m2 (merge2 l2) (g2t r2)
          | otherwise        = binA p2 m2 (g2t l2) (merge2 r2)
        merge2 (NE.Tip k2 x2) = mergeTips k1 x1 k2 x2

    go' t1' (NE.Tip k2 x2) = merge1 t1'
      where
        merge1 t1@(NE.Bin p1 m1 l1 r1)
          | nomatch k2 p1 m1 = linkA p1 (g1t t1) k2 (subsingletonBy g2k k2 x2)
          | zero k2 m1       = binA p1 m1 (merge1 l1) (g1t r1)
          | otherwise        = binA p1 m1 (g1t l1) (merge1 r1)
        merge1 (NE.Tip k1 x1) = mergeTips k1 x1 k2 x2

    go' t1@(NE.Bin p1 m1 l1 r1) t2@(NE.Bin p2 m2 l2 r2)
      | shorter m1 m2  = merge1
      | shorter m2 m1  = merge2
      | p1 == p2       = binA p1 m1 (go' l1 l2) (go' r1 r2)
      | otherwise      = linkA p1 (g1t t1) p2 (g2t t2)
      where
        merge1 | nomatch p2 p1 m1  = linkA p1 (g1t t1) p2 (g2t t2)
               | zero p2 m1        = binA p1 m1 (go'  l1 t2) (g1t r1)
               | otherwise         = binA p1 m1 (g1t l1)    (go'  r1 t2)
        merge2 | nomatch p1 p2 m2  = linkA p1 (g1t t1) p2 (g2t t2)
               | zero p1 m2        = binA p2 m2 (go'  t1 l2) (g2t    r2)
               | otherwise         = binA p2 m2 (g2t    l2) (go'  t1 r2)

    subsingletonBy gk k x = maybe Nil (NonEmpty . NE.Tip k) <$> gk k x
    {-# INLINE subsingletonBy #-}

    mergeTips k1 x1 k2 x2
      | k1 == k2  = maybe Nil (NonEmpty . NE.Tip k1) <$> f k1 x1 x2
      | k1 <  k2  = liftA2 (subdoubleton k1 k2) (g1k k1 x1) (g2k k2 x2)
        {-
        = link_ k1 k2 <$> subsingletonBy g1k k1 x1 <*> subsingletonBy g2k k2 x2
        -}
      | otherwise = liftA2 (subdoubleton k2 k1) (g2k k2 x2) (g1k k1 x1)
    {-# INLINE mergeTips #-}

    subdoubleton _ _   Nothing Nothing     = Nil
    subdoubleton _ k2  Nothing (Just y2)   = NonEmpty (NE.Tip k2 y2)
    subdoubleton k1 _  (Just y1) Nothing   = NonEmpty (NE.Tip k1 y1)
    subdoubleton k1 k2 (Just y1) (Just y2) = NonEmpty $ NE.link k1 (NE.Tip k1 y1) k2 (NE.Tip k2 y2)
    {-# INLINE subdoubleton #-}

    -- | A variant of 'link_' which makes sure to execute side-effects
    -- in the right order.
    linkA
        :: Applicative f
        => Prefix -> f (IntMap x)
        -> Prefix -> f (IntMap x)
        -> f (IntMap x)
    linkA p1 t1 p2 t2
      | zero p1 m = binA p m t1 t2
      | otherwise = binA p m t2 t1
      where
        m = branchMask p1 p2
        p = mask p1 m
    {-# INLINE linkA #-}

    -- A variant of 'bin' that ensures that effects for negative keys are executed
    -- first.
    binA
        :: Applicative f
        => Prefix
        -> Mask
        -> f (IntMap x)
        -> f (IntMap x)
        -> f (IntMap x)
    binA p m a b
      | m < 0     = liftA2 (flip (bin p m)) b a
      | otherwise = liftA2       (bin p m)  a b
    {-# INLINE binA #-}
{-# INLINE mergeA #-}


{--------------------------------------------------------------------
  Min\/Max
--------------------------------------------------------------------}

-- | \(O(\min(n,W))\). Update the value at the minimal key.
--
-- > updateMinWithKey (\ k a -> Just ((show k) ++ ":" ++ a)) (fromList [(5,"a"), (3,"b")]) == fromList [(3,"3:b"), (5,"a")]
-- > updateMinWithKey (\ _ _ -> Nothing)                     (fromList [(5,"a"), (3,"b")]) == singleton 5 "a"

updateMinWithKey :: (Key -> a -> Maybe a) -> IntMap a -> IntMap a
updateMinWithKey f (NonEmpty ne)
  | NE.Bin p m l r <- ne
  , m < 0
  = binCheckRight p m l (go f r)
  | otherwise
  = go f ne
  where
    go f' (NE.Bin p m l r) = binCheckLeft p m (go f' l) r
    go f' (NE.Tip k y) = case f' k y of
                        Just y' -> NonEmpty (NE.Tip k y')
                        Nothing -> Nil
updateMinWithKey _ Nil = error "updateMinWithKey Nil"

-- | \(O(\min(n,W))\). Update the value at the maximal key.
--
-- > updateMaxWithKey (\ k a -> Just ((show k) ++ ":" ++ a)) (fromList [(5,"a"), (3,"b")]) == fromList [(3,"b"), (5,"5:a")]
-- > updateMaxWithKey (\ _ _ -> Nothing)                     (fromList [(5,"a"), (3,"b")]) == singleton 3 "b"

updateMaxWithKey :: (Key -> a -> Maybe a) -> IntMap a -> IntMap a
updateMaxWithKey f (NonEmpty ne)
  | NE.Bin p m l r <- ne
  , m < 0
  = binCheckLeft p m (go f l) r
  | otherwise
  = go f ne

  where
    go f' (NE.Bin p m l r) = binCheckRight p m l (go f' r)
    go f' (NE.Tip k y) = case f' k y of
                        Just y' -> NonEmpty (NE.Tip k y')
                        Nothing -> Nil
updateMaxWithKey _ Nil = error "updateMaxWithKey Nil"

data View a = View {-# UNPACK #-} !Key a !(IntMap a)

-- | \(O(\min(n,W))\). Retrieves the maximal (key,value) pair of the map, and
-- the map stripped of that element, or 'Nothing' if passed an empty map.
--
-- > maxViewWithKey (fromList [(5,"a"), (3,"b")]) == Just ((5,"a"), singleton 3 "b")
-- > maxViewWithKey empty == Nothing

maxViewWithKey :: IntMap a -> Maybe ((Key, a), IntMap a)
maxViewWithKey t = case t of
  Nil -> Nothing
  _ -> Just $ case maxViewWithKeySure t of
                View k v t' -> ((k, v), t')
{-# INLINE maxViewWithKey #-}

maxViewWithKeySure :: IntMap a -> View a
maxViewWithKeySure (NonEmpty t)
  | NE.Bin p m l r <- t
  , m < 0
  = case go l of View k a l' -> View k a (binCheckLeft p m l' r)
  | otherwise
  = go t
  where
    go (NE.Bin p m l r) =
        case go r of View k a r' -> View k a (binCheckRight p m l r')
    go (NE.Tip k y) = View k y Nil
maxViewWithKeySure Nil = error "maxViewWithKeySure Nil"
-- See note on NOINLINE at minViewWithKeySure
{-# NOINLINE maxViewWithKeySure #-}

-- | \(O(\min(n,W))\). Retrieves the minimal (key,value) pair of the map, and
-- the map stripped of that element, or 'Nothing' if passed an empty map.
--
-- > minViewWithKey (fromList [(5,"a"), (3,"b")]) == Just ((3,"b"), singleton 5 "a")
-- > minViewWithKey empty == Nothing

minViewWithKey :: IntMap a -> Maybe ((Key, a), IntMap a)
minViewWithKey t =
  case t of
    Nil -> Nothing
    _ -> Just $ case minViewWithKeySure t of
                  View k v t' -> ((k, v), t')
-- We inline this to give GHC the best possible chance of
-- getting rid of the Maybe, pair, and Int constructors, as
-- well as a thunk under the Just. That is, we really want to
-- be certain this inlines!
{-# INLINE minViewWithKey #-}

minViewWithKeySure :: IntMap a -> View a
minViewWithKeySure (NonEmpty t)
  | NE.Bin p m l r <- t
  , m < 0
  = case go r of
        View k a r' -> View k a (binCheckRight p m l r')
  | otherwise
  = go t
  where
    go (NE.Bin p m l r) =
        case go l of View k a l' -> View k a (binCheckLeft p m l' r)
    go (NE.Tip k y) = View k y Nil
minViewWithKeySure Nil = error "minViewWithKeySure Nil"
-- There's never anything significant to be gained by inlining
-- this. Sufficiently recent GHC versions will inline the wrapper
-- anyway, which should be good enough.
{-# NOINLINE minViewWithKeySure #-}

-- | \(O(\min(n,W))\). Update the value at the maximal key.
--
-- > updateMax (\ a -> Just ("X" ++ a)) (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "Xa")]
-- > updateMax (\ _ -> Nothing)         (fromList [(5,"a"), (3,"b")]) == singleton 3 "b"

updateMax :: (a -> Maybe a) -> IntMap a -> IntMap a
updateMax f = updateMaxWithKey (const f)

-- | \(O(\min(n,W))\). Update the value at the minimal key.
--
-- > updateMin (\ a -> Just ("X" ++ a)) (fromList [(5,"a"), (3,"b")]) == fromList [(3, "Xb"), (5, "a")]
-- > updateMin (\ _ -> Nothing)         (fromList [(5,"a"), (3,"b")]) == singleton 5 "a"

updateMin :: (a -> Maybe a) -> IntMap a -> IntMap a
updateMin f = updateMinWithKey (const f)

-- | \(O(\min(n,W))\). Retrieves the maximal key of the map, and the map
-- stripped of that element, or 'Nothing' if passed an empty map.
maxView :: IntMap a -> Maybe (a, IntMap a)
maxView t = fmap (\((_, x), t') -> (x, t')) (maxViewWithKey t)

-- | \(O(\min(n,W))\). Retrieves the minimal key of the map, and the map
-- stripped of that element, or 'Nothing' if passed an empty map.
minView :: IntMap a -> Maybe (a, IntMap a)
minView t = fmap (\((_, x), t') -> (x, t')) (minViewWithKey t)

-- | \(O(\min(n,W))\). Delete and find the maximal element.
-- This function throws an error if the map is empty. Use 'maxViewWithKey'
-- if the map may be empty.
deleteFindMax :: IntMap a -> ((Key, a), IntMap a)
deleteFindMax = fromMaybe (error "deleteFindMax: empty map has no maximal element") . maxViewWithKey

-- | \(O(\min(n,W))\). Delete and find the minimal element.
-- This function throws an error if the map is empty. Use 'minViewWithKey'
-- if the map may be empty.
deleteFindMin :: IntMap a -> ((Key, a), IntMap a)
deleteFindMin = fromMaybe (error "deleteFindMin: empty map has no minimal element") . minViewWithKey

-- | \(O(\min(n,W))\). The minimal key of the map. Returns 'Nothing' if the map is empty.
lookupMin :: IntMap a -> Maybe (Key, a)
lookupMin Nil = Nothing
lookupMin (NonEmpty (NE.Tip k v)) = Just (k,v)
lookupMin (NonEmpty (NE.Bin _ m l r))
  | m < 0     = go r
  | otherwise = go l
    where go (NE.Tip k v)      = Just (k,v)
          go (NE.Bin _ _ l' _) = go l'

-- | \(O(\min(n,W))\). The minimal key of the map. Calls 'error' if the map is empty.
-- Use 'minViewWithKey' if the map may be empty.
findMin :: IntMap a -> (Key, a)
findMin t
  | Just r <- lookupMin t = r
  | otherwise = error "findMin: empty map has no minimal element"

-- | \(O(\min(n,W))\). The maximal key of the map. Returns 'Nothing' if the map is empty.
lookupMax :: IntMap a -> Maybe (Key, a)
lookupMax Nil = Nothing
lookupMax (NonEmpty (NE.Tip k v)) = Just (k,v)
lookupMax (NonEmpty (NE.Bin _ m l r))
  | m < 0     = go l
  | otherwise = go r
    where go (NE.Tip k v)      = Just (k,v)
          go (NE.Bin _ _ _ r') = go r'

-- | \(O(\min(n,W))\). The maximal key of the map. Calls 'error' if the map is empty.
-- Use 'maxViewWithKey' if the map may be empty.
findMax :: IntMap a -> (Key, a)
findMax t
  | Just r <- lookupMax t = r
  | otherwise = error "findMax: empty map has no maximal element"

-- | \(O(\min(n,W))\). Delete the minimal key. Returns an empty map if the map is empty.
--
-- Note that this is a change of behaviour for consistency with 'Data.Map.Map' &#8211;
-- versions prior to 0.5 threw an error if the 'IntMap' was already empty.
deleteMin :: IntMap a -> IntMap a
deleteMin = maybe Nil snd . minView

-- | \(O(\min(n,W))\). Delete the maximal key. Returns an empty map if the map is empty.
--
-- Note that this is a change of behaviour for consistency with 'Data.Map.Map' &#8211;
-- versions prior to 0.5 threw an error if the 'IntMap' was already empty.
deleteMax :: IntMap a -> IntMap a
deleteMax = maybe Nil snd . maxView


{--------------------------------------------------------------------
  Submap
--------------------------------------------------------------------}
-- | \(O(n+m)\). Is this a proper submap? (ie. a submap but not equal).
-- Defined as (@'isProperSubmapOf' = 'isProperSubmapOfBy' (==)@).
isProperSubmapOf :: Eq a => IntMap a -> IntMap a -> Bool
isProperSubmapOf m1 m2
  = isProperSubmapOfBy (==) m1 m2

{- | \(O(n+m)\). Is this a proper submap? (ie. a submap but not equal).
 The expression (@'isProperSubmapOfBy' f m1 m2@) returns 'True' when
 @keys m1@ and @keys m2@ are not equal,
 all keys in @m1@ are in @m2@, and when @f@ returns 'True' when
 applied to their respective values. For example, the following
 expressions are all 'True':

  > isProperSubmapOfBy (==) (fromList [(1,1)]) (fromList [(1,1),(2,2)])
  > isProperSubmapOfBy (<=) (fromList [(1,1)]) (fromList [(1,1),(2,2)])

 But the following are all 'False':

  > isProperSubmapOfBy (==) (fromList [(1,1),(2,2)]) (fromList [(1,1),(2,2)])
  > isProperSubmapOfBy (==) (fromList [(1,1),(2,2)]) (fromList [(1,1)])
  > isProperSubmapOfBy (<)  (fromList [(1,1)])       (fromList [(1,1),(2,2)])
-}
isProperSubmapOfBy :: (a -> b -> Bool) -> IntMap a -> IntMap b -> Bool
isProperSubmapOfBy f (NonEmpty t1) (NonEmpty t2) = NE.isProperSubmapOfBy f t1 t2
isProperSubmapOfBy _ _ Nil = False
isProperSubmapOfBy _ Nil _ = True


-- | \(O(n+m)\). Is this a submap?
-- Defined as (@'isSubmapOf' = 'isSubmapOfBy' (==)@).
isSubmapOf :: Eq a => IntMap a -> IntMap a -> Bool
isSubmapOf m1 m2
  = isSubmapOfBy (==) m1 m2

{- | \(O(n+m)\).
 The expression (@'isSubmapOfBy' f m1 m2@) returns 'True' if
 all keys in @m1@ are in @m2@, and when @f@ returns 'True' when
 applied to their respective values. For example, the following
 expressions are all 'True':

  > isSubmapOfBy (==) (fromList [(1,1)]) (fromList [(1,1),(2,2)])
  > isSubmapOfBy (<=) (fromList [(1,1)]) (fromList [(1,1),(2,2)])
  > isSubmapOfBy (==) (fromList [(1,1),(2,2)]) (fromList [(1,1),(2,2)])

 But the following are all 'False':

  > isSubmapOfBy (==) (fromList [(1,2)]) (fromList [(1,1),(2,2)])
  > isSubmapOfBy (<) (fromList [(1,1)]) (fromList [(1,1),(2,2)])
  > isSubmapOfBy (==) (fromList [(1,1),(2,2)]) (fromList [(1,1)])
-}
isSubmapOfBy :: (a -> b -> Bool) -> IntMap a -> IntMap b -> Bool
isSubmapOfBy f (NonEmpty t1) (NonEmpty t2) = NE.isSubmapOfBy f t1 t2
isSubmapOfBy _ Nil _ = True
isSubmapOfBy _ _ Nil = False

{--------------------------------------------------------------------
  Mapping
--------------------------------------------------------------------}
-- | \(O(n)\). Map a function over all values in the map.
--
-- > map (++ "x") (fromList [(5,"a"), (3,"b")]) == fromList [(3, "bx"), (5, "ax")]

map :: (a -> b) -> IntMap a -> IntMap b
map f (NonEmpty m) = NonEmpty (NE.map f m)
map _ Nil = Nil

#ifdef __GLASGOW_HASKELL__
{-# NOINLINE [1] map #-}
{-# RULES
"map/map" forall f g xs . map f (map g xs) = map (f . g) xs
"map/coerce" map coerce = coerce
 #-}
#endif

-- | \(O(n)\). Map a function over all values in the map.
--
-- > let f key x = (show key) ++ ":" ++ x
-- > mapWithKey f (fromList [(5,"a"), (3,"b")]) == fromList [(3, "3:b"), (5, "5:a")]

mapWithKey :: (Key -> a -> b) -> IntMap a -> IntMap b
mapWithKey f (NonEmpty m) = NonEmpty $ NE.mapWithKey f m
mapWithKey _ Nil = Nil

#ifdef __GLASGOW_HASKELL__
{-# NOINLINE [1] mapWithKey #-}
{-# RULES
"mapWithKey/mapWithKey" forall f g xs . mapWithKey f (mapWithKey g xs) =
  mapWithKey (\k a -> f k (g k a)) xs
"mapWithKey/map" forall f g xs . mapWithKey f (map g xs) =
  mapWithKey (\k a -> f k (g a)) xs
"map/mapWithKey" forall f g xs . map f (mapWithKey g xs) =
  mapWithKey (\k a -> f (g k a)) xs
 #-}
#endif

-- | \(O(n)\).
-- @'traverseWithKey' f s == 'fromList' <$> 'traverse' (\(k, v) -> (,) k <$> f k v) ('toList' m)@
-- That is, behaves exactly like a regular 'traverse' except that the traversing
-- function also has access to the key associated with a value.
--
-- > traverseWithKey (\k v -> if odd k then Just (succ v) else Nothing) (fromList [(1, 'a'), (5, 'e')]) == Just (fromList [(1, 'b'), (5, 'f')])
-- > traverseWithKey (\k v -> if odd k then Just (succ v) else Nothing) (fromList [(2, 'c')])           == Nothing
traverseWithKey :: Applicative t => (Key -> a -> t b) -> IntMap a -> t (IntMap b)
traverseWithKey f = \ t -> case t of
  NonEmpty m -> NonEmpty <$> NE.traverseWithKey f m
  Nil        -> pure Nil
{-# INLINE traverseWithKey #-}

-- | \(O(n)\). The function @'mapAccum'@ threads an accumulating
-- argument through the map in ascending order of keys.
--
-- > let f a b = (a ++ b, b ++ "X")
-- > mapAccum f "Everything: " (fromList [(5,"a"), (3,"b")]) == ("Everything: ba", fromList [(3, "bX"), (5, "aX")])

mapAccum :: (a -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccum f = mapAccumWithKey (\a' _ x -> f a' x)

-- | \(O(n)\). The function @'mapAccumWithKey'@ threads an accumulating
-- argument through the map in ascending order of keys.
--
-- > let f a k b = (a ++ " " ++ (show k) ++ "-" ++ b, b ++ "X")
-- > mapAccumWithKey f "Everything:" (fromList [(5,"a"), (3,"b")]) == ("Everything: 3-b 5-a", fromList [(3, "bX"), (5, "aX")])

mapAccumWithKey :: (a -> Key -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccumWithKey f a t
  = mapAccumL f a t

-- | \(O(n)\). The function @'mapAccumL'@ threads an accumulating
-- argument through the map in ascending order of keys.
mapAccumL :: (a -> Key -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccumL f a t
  = case t of
      NonEmpty m -> fmap NonEmpty $ NE.mapAccumL f a m
      Nil        -> (a,Nil)

-- | \(O(n)\). The function @'mapAccumRWithKey'@ threads an accumulating
-- argument through the map in descending order of keys.
mapAccumRWithKey :: (a -> Key -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccumRWithKey f a t
  = case t of
      NonEmpty m -> fmap NonEmpty $ NE.mapAccumRWithKey f a m
      Nil        -> (a,Nil)

-- | \(O(n \min(n,W))\).
-- @'mapKeys' f s@ is the map obtained by applying @f@ to each key of @s@.
--
-- The size of the result may be smaller if @f@ maps two or more distinct
-- keys to the same new key.  In this case the value at the greatest of the
-- original keys is retained.
--
-- > mapKeys (+ 1) (fromList [(5,"a"), (3,"b")])                        == fromList [(4, "b"), (6, "a")]
-- > mapKeys (\ _ -> 1) (fromList [(1,"b"), (2,"a"), (3,"d"), (4,"c")]) == singleton 1 "c"
-- > mapKeys (\ _ -> 3) (fromList [(1,"b"), (2,"a"), (3,"d"), (4,"c")]) == singleton 3 "c"

mapKeys :: (Key->Key) -> IntMap a -> IntMap a
mapKeys f = fromList . foldrWithKey (\k x xs -> (f k, x) : xs) []

-- | \(O(n \min(n,W))\).
-- @'mapKeysWith' c f s@ is the map obtained by applying @f@ to each key of @s@.
--
-- The size of the result may be smaller if @f@ maps two or more distinct
-- keys to the same new key.  In this case the associated values will be
-- combined using @c@.
--
-- > mapKeysWith (++) (\ _ -> 1) (fromList [(1,"b"), (2,"a"), (3,"d"), (4,"c")]) == singleton 1 "cdab"
-- > mapKeysWith (++) (\ _ -> 3) (fromList [(1,"b"), (2,"a"), (3,"d"), (4,"c")]) == singleton 3 "cdab"

mapKeysWith :: (a -> a -> a) -> (Key->Key) -> IntMap a -> IntMap a
mapKeysWith c f
  = fromListWith c . foldrWithKey (\k x xs -> (f k, x) : xs) []

-- | \(O(n \min(n,W))\).
-- @'mapKeysMonotonic' f s == 'mapKeys' f s@, but works only when @f@
-- is strictly monotonic.
-- That is, for any values @x@ and @y@, if @x@ < @y@ then @f x@ < @f y@.
-- /The precondition is not checked./
-- Semi-formally, we have:
--
-- > and [x < y ==> f x < f y | x <- ls, y <- ls]
-- >                     ==> mapKeysMonotonic f s == mapKeys f s
-- >     where ls = keys s
--
-- This means that @f@ maps distinct original keys to distinct resulting keys.
-- This function has slightly better performance than 'mapKeys'.
--
-- > mapKeysMonotonic (\ k -> k * 2) (fromList [(5,"a"), (3,"b")]) == fromList [(6, "b"), (10, "a")]

mapKeysMonotonic :: (Key->Key) -> IntMap a -> IntMap a
mapKeysMonotonic f
  = fromDistinctAscList . foldrWithKey (\k x xs -> (f k, x) : xs) []

{--------------------------------------------------------------------
  Filter
--------------------------------------------------------------------}
-- | \(O(n)\). Filter all values that satisfy some predicate.
--
-- > filter (> "a") (fromList [(5,"a"), (3,"b")]) == singleton 3 "b"
-- > filter (> "x") (fromList [(5,"a"), (3,"b")]) == empty
-- > filter (< "a") (fromList [(5,"a"), (3,"b")]) == empty

filter :: (a -> Bool) -> IntMap a -> IntMap a
filter p m
  = filterWithKey (\_ x -> p x) m

-- | \(O(n)\). Filter all keys\/values that satisfy some predicate.
--
-- > filterWithKey (\k _ -> k > 4) (fromList [(5,"a"), (3,"b")]) == singleton 5 "a"

filterWithKey :: (Key -> a -> Bool) -> IntMap a -> IntMap a
filterWithKey predicate = go
    where
    go Nil          = Nil
    go (NonEmpty t) = go' t
    go' t@(NE.Tip k x) = if predicate k x then NonEmpty t else Nil
    go' (NE.Bin p m l r) = bin p m (go' l) (go' r)

-- | \(O(n)\). Partition the map according to some predicate. The first
-- map contains all elements that satisfy the predicate, the second all
-- elements that fail the predicate. See also 'split'.
--
-- > partition (> "a") (fromList [(5,"a"), (3,"b")]) == (singleton 3 "b", singleton 5 "a")
-- > partition (< "x") (fromList [(5,"a"), (3,"b")]) == (fromList [(3, "b"), (5, "a")], empty)
-- > partition (> "x") (fromList [(5,"a"), (3,"b")]) == (empty, fromList [(3, "b"), (5, "a")])

partition :: (a -> Bool) -> IntMap a -> (IntMap a,IntMap a)
partition p m
  = partitionWithKey (\_ x -> p x) m

-- | \(O(n)\). Partition the map according to some predicate. The first
-- map contains all elements that satisfy the predicate, the second all
-- elements that fail the predicate. See also 'split'.
--
-- > partitionWithKey (\ k _ -> k > 3) (fromList [(5,"a"), (3,"b")]) == (singleton 5 "a", singleton 3 "b")
-- > partitionWithKey (\ k _ -> k < 7) (fromList [(5,"a"), (3,"b")]) == (fromList [(3, "b"), (5, "a")], empty)
-- > partitionWithKey (\ k _ -> k > 7) (fromList [(5,"a"), (3,"b")]) == (empty, fromList [(3, "b"), (5, "a")])

partitionWithKey :: (Key -> a -> Bool) -> IntMap a -> (IntMap a,IntMap a)
partitionWithKey predicate t0 = toPair $ go t0
  where
    go (NonEmpty m) = go' m
    go Nil = Nil :*: Nil
    go' (NE.Bin p m l r)
      = let (l1 :*: l2) = go' l
            (r1 :*: r2) = go' r
        in bin p m l1 r1 :*: bin p m l2 r2
    go' t@(NE.Tip k x)
      | predicate k x = (NonEmpty t :*: Nil)
      | otherwise     = (Nil :*: NonEmpty t)

-- | \(O(\min(n,W))\). Take while a predicate on the keys holds.
-- The user is responsible for ensuring that for all @Int@s, @j \< k ==\> p j \>= p k@.
-- See note at 'spanAntitone'.
--
-- @
-- takeWhileAntitone p = 'fromDistinctAscList' . 'Data.List.takeWhile' (p . fst) . 'toList'
-- takeWhileAntitone p = 'filterWithKey' (\\k _ -> p k)
-- @
--
-- @since 0.6.7
takeWhileAntitone :: (Key -> Bool) -> IntMap a -> IntMap a
takeWhileAntitone predicate t =
  case t of
    NonEmpty (NE.Bin p m l r)
      | m < 0 ->
        if predicate 0 -- handle negative numbers.
        then binCheckLeft p m (go' l) r
        else go' r
    _ -> go t
  where
    go (NonEmpty m) = go' m
    go Nil          = Nil
    go' (NE.Bin p m l r)
      | predicate $! p+m = binCheckRight p m l (go' r)
      | otherwise        = go' l
    go' t'@(NE.Tip ky _)
      | predicate ky = NonEmpty t'
      | otherwise    = Nil

-- | \(O(\min(n,W))\). Drop while a predicate on the keys holds.
-- The user is responsible for ensuring that for all @Int@s, @j \< k ==\> p j \>= p k@.
-- See note at 'spanAntitone'.
--
-- @
-- dropWhileAntitone p = 'fromDistinctAscList' . 'Data.List.dropWhile' (p . fst) . 'toList'
-- dropWhileAntitone p = 'filterWithKey' (\\k _ -> not (p k))
-- @
--
-- @since 0.6.7
dropWhileAntitone :: (Key -> Bool) -> IntMap a -> IntMap a
dropWhileAntitone predicate t =
  case t of
    NonEmpty (NE.Bin p m l r)
      | m < 0 ->
        if predicate 0 -- handle negative numbers.
        then go' l
        else binCheckRight p m l (go' r)
    _ -> go t
  where
    go (NonEmpty m) = go' m
    go Nil = Nil
    go' (NE.Bin p m l r)
      | predicate $! p+m = go' r
      | otherwise        = binCheckLeft p m (go' l) r
    go' t'@(NE.Tip ky _)
      | predicate ky = Nil
      | otherwise    = NonEmpty t'

-- | \(O(\min(n,W))\). Divide a map at the point where a predicate on the keys stops holding.
-- The user is responsible for ensuring that for all @Int@s, @j \< k ==\> p j \>= p k@.
--
-- @
-- spanAntitone p xs = ('takeWhileAntitone' p xs, 'dropWhileAntitone' p xs)
-- spanAntitone p xs = 'partitionWithKey' (\\k _ -> p k) xs
-- @
--
-- Note: if @p@ is not actually antitone, then @spanAntitone@ will split the map
-- at some /unspecified/ point.
--
-- @since 0.6.7
spanAntitone :: (Key -> Bool) -> IntMap a -> (IntMap a, IntMap a)
spanAntitone predicate t =
  case t of
    NonEmpty (NE.Bin p m l r)
      | m < 0 ->
        if predicate 0 -- handle negative numbers.
        then
          case go' l of
            (lt :*: gt) ->
              let !lt' = binCheckLeft p m lt r
              in (lt', gt)
        else
          case go' r of
            (lt :*: gt) ->
              let !gt' = binCheckRight p m l gt
              in (lt, gt')
    _ -> case go t of
          (lt :*: gt) -> (lt, gt)
  where
    go (NonEmpty m) = go' m
    go Nil = (Nil :*: Nil)
    go' (NE.Bin p m l r)
      | predicate $! p+m = case go' r of (lt :*: gt) -> binCheckRight p m l lt :*: gt
      | otherwise        = case go' l of (lt :*: gt) -> lt :*: binCheckLeft p m gt r
    go' t'@(NE.Tip ky _)
      | predicate ky = (NonEmpty t' :*: Nil)
      | otherwise    = (Nil :*: NonEmpty t')

-- | \(O(n)\). Map values and collect the 'Just' results.
--
-- > let f x = if x == "a" then Just "new a" else Nothing
-- > mapMaybe f (fromList [(5,"a"), (3,"b")]) == singleton 5 "new a"

mapMaybe :: (a -> Maybe b) -> IntMap a -> IntMap b
mapMaybe f = mapMaybeWithKey (\_ x -> f x)

-- | \(O(n)\). Map keys\/values and collect the 'Just' results.
--
-- > let f k _ = if k < 5 then Just ("key : " ++ (show k)) else Nothing
-- > mapMaybeWithKey f (fromList [(5,"a"), (3,"b")]) == singleton 3 "key : 3"

mapMaybeWithKey :: (Key -> a -> Maybe b) -> IntMap a -> IntMap b
mapMaybeWithKey f (NonEmpty t) = go t
  where
    go (NE.Bin p m l r) = bin p m (go l) (go r)
    go (NE.Tip k x) = case f k x of
      Just y -> NonEmpty (NE.Tip k y)
      _      -> Nil
mapMaybeWithKey _ Nil = Nil

-- | \(O(n)\). Map values and separate the 'Left' and 'Right' results.
--
-- > let f a = if a < "c" then Left a else Right a
-- > mapEither f (fromList [(5,"a"), (3,"b"), (1,"x"), (7,"z")])
-- >     == (fromList [(3,"b"), (5,"a")], fromList [(1,"x"), (7,"z")])
-- >
-- > mapEither (\ a -> Right a) (fromList [(5,"a"), (3,"b"), (1,"x"), (7,"z")])
-- >     == (empty, fromList [(5,"a"), (3,"b"), (1,"x"), (7,"z")])

mapEither :: (a -> Either b c) -> IntMap a -> (IntMap b, IntMap c)
mapEither f m
  = mapEitherWithKey (\_ x -> f x) m

-- | \(O(n)\). Map keys\/values and separate the 'Left' and 'Right' results.
--
-- > let f k a = if k < 5 then Left (k * 2) else Right (a ++ a)
-- > mapEitherWithKey f (fromList [(5,"a"), (3,"b"), (1,"x"), (7,"z")])
-- >     == (fromList [(1,2), (3,6)], fromList [(5,"aa"), (7,"zz")])
-- >
-- > mapEitherWithKey (\_ a -> Right a) (fromList [(5,"a"), (3,"b"), (1,"x"), (7,"z")])
-- >     == (empty, fromList [(1,"x"), (3,"b"), (5,"a"), (7,"z")])

mapEitherWithKey :: (Key -> a -> Either b c) -> IntMap a -> (IntMap b, IntMap c)
mapEitherWithKey f0 (NonEmpty t0) = toPair $ go f0 t0
  where
    go f (NE.Bin p m l r) =
      bin p m l1 r1 :*: bin p m l2 r2
      where
        (l1 :*: l2) = go f l
        (r1 :*: r2) = go f r
    go f (NE.Tip k x) = case f k x of
      Left y  -> (NonEmpty (NE.Tip k y) :*: Nil)
      Right z -> (Nil :*: NonEmpty (NE.Tip k z))
mapEitherWithKey _ Nil = (Nil, Nil)

-- | \(O(\min(n,W))\). The expression (@'split' k map@) is a pair @(map1,map2)@
-- where all keys in @map1@ are lower than @k@ and all keys in
-- @map2@ larger than @k@. Any key equal to @k@ is found in neither @map1@ nor @map2@.
--
-- > split 2 (fromList [(5,"a"), (3,"b")]) == (empty, fromList [(3,"b"), (5,"a")])
-- > split 3 (fromList [(5,"a"), (3,"b")]) == (empty, singleton 5 "a")
-- > split 4 (fromList [(5,"a"), (3,"b")]) == (singleton 3 "b", singleton 5 "a")
-- > split 5 (fromList [(5,"a"), (3,"b")]) == (singleton 3 "b", empty)
-- > split 6 (fromList [(5,"a"), (3,"b")]) == (fromList [(3,"b"), (5,"a")], empty)

split :: Key -> IntMap a -> (IntMap a, IntMap a)
split k (NonEmpty t) =
  case t of
    NE.Bin p m l r
      | m < 0 ->
        if k >= 0 -- handle negative numbers.
        then
          case go k l of
            (lt :*: gt) ->
              let !lt' = binCheckLeft p m lt r
              in (lt', gt)
        else
          case go k r of
            (lt :*: gt) ->
              let !gt' = binCheckRight p m l gt
              in (lt, gt')
    _ -> case go k t of
          (lt :*: gt) -> (lt, gt)
  where
    go :: Key -> NE.IntMap a -> StrictPair (IntMap a) (IntMap a)
    go k' t'@(NE.Bin p m l r)
      | nomatch k' p m = if k' > p then NonEmpty t' :*: Nil else Nil :*: NonEmpty t'
      | zero k' m = case go k' l of (lt :*: gt) -> lt :*: binCheckLeft p m gt r
      | otherwise = case go k' r of (lt :*: gt) -> binCheckRight p m l lt :*: gt
    go k' t'@(NE.Tip ky _)
      | k' > ky   = (NonEmpty t' :*: Nil)
      | k' < ky   = (Nil :*: NonEmpty t')
      | otherwise = (Nil :*: Nil)
split _ Nil = (Nil, Nil)

data SplitLookup a = SplitLookup !(IntMap a) !(Maybe a) !(IntMap a)

mapLT :: (IntMap a -> IntMap a) -> SplitLookup a -> SplitLookup a
mapLT f (SplitLookup lt fnd gt) = SplitLookup (f lt) fnd gt
{-# INLINE mapLT #-}

mapGT :: (IntMap a -> IntMap a) -> SplitLookup a -> SplitLookup a
mapGT f (SplitLookup lt fnd gt) = SplitLookup lt fnd (f gt)
{-# INLINE mapGT #-}

-- | \(O(\min(n,W))\). Performs a 'split' but also returns whether the pivot
-- key was found in the original map.
--
-- > splitLookup 2 (fromList [(5,"a"), (3,"b")]) == (empty, Nothing, fromList [(3,"b"), (5,"a")])
-- > splitLookup 3 (fromList [(5,"a"), (3,"b")]) == (empty, Just "b", singleton 5 "a")
-- > splitLookup 4 (fromList [(5,"a"), (3,"b")]) == (singleton 3 "b", Nothing, singleton 5 "a")
-- > splitLookup 5 (fromList [(5,"a"), (3,"b")]) == (singleton 3 "b", Just "a", empty)
-- > splitLookup 6 (fromList [(5,"a"), (3,"b")]) == (fromList [(3,"b"), (5,"a")], Nothing, empty)

splitLookup :: Key -> IntMap a -> (IntMap a, Maybe a, IntMap a)
splitLookup k (NonEmpty t) =
  case
    case t of
      NE.Bin p m l r
        | m < 0 ->
          if k >= 0 -- handle negative numbers.
          then mapLT (flip (binCheckLeft p m) r) (go k l)
          else mapGT (binCheckRight p m l) (go k r)
      _ -> go k t
  of SplitLookup lt fnd gt -> (lt, fnd, gt)
  where
    go k' t'@(NE.Bin p m l r)
      | nomatch k' p m =
          if k' > p
          then SplitLookup (NonEmpty t') Nothing Nil
          else SplitLookup Nil Nothing (NonEmpty t')
      | zero k' m = mapGT (flip (binCheckLeft p m) r) (go k' l)
      | otherwise = mapLT (binCheckRight p m l) (go k' r)
    go k' t'@(NE.Tip ky y)
      | k' > ky   = SplitLookup (NonEmpty t')  Nothing  Nil
      | k' < ky   = SplitLookup Nil Nothing  (NonEmpty t')
      | otherwise = SplitLookup Nil (Just y) Nil
splitLookup _ Nil = (Nil, Nothing, Nil)

{--------------------------------------------------------------------
  Fold
--------------------------------------------------------------------}
-- | \(O(n)\). Fold the values in the map using the given right-associative
-- binary operator, such that @'foldr' f z == 'Prelude.foldr' f z . 'elems'@.
--
-- For example,
--
-- > elems map = foldr (:) [] map
--
-- > let f a len = len + (length a)
-- > foldr f 0 (fromList [(5,"a"), (3,"bbb")]) == 4
foldr :: (a -> b -> b) -> b -> IntMap a -> b
foldr f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    NonEmpty m -> NE.foldr f z m
    _ -> z
{-# INLINE foldr #-}

-- | \(O(n)\). A strict version of 'foldr'. Each application of the operator is
-- evaluated before using the result in the next application. This
-- function is strict in the starting value.
foldr' :: (a -> b -> b) -> b -> IntMap a -> b
foldr' f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    NonEmpty m -> NE.foldr' f z m
    _ -> z
{-# INLINE foldr' #-}

-- | \(O(n)\). Fold the values in the map using the given left-associative
-- binary operator, such that @'foldl' f z == 'Prelude.foldl' f z . 'elems'@.
--
-- For example,
--
-- > elems = reverse . foldl (flip (:)) []
--
-- > let f len a = len + (length a)
-- > foldl f 0 (fromList [(5,"a"), (3,"bbb")]) == 4
foldl :: (a -> b -> a) -> a -> IntMap b -> a
foldl f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    NonEmpty m -> NE.foldl f z m
    _ -> z
{-# INLINE foldl #-}

-- | \(O(n)\). A strict version of 'foldl'. Each application of the operator is
-- evaluated before using the result in the next application. This
-- function is strict in the starting value.
foldl' :: (a -> b -> a) -> a -> IntMap b -> a
foldl' f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    NonEmpty m -> NE.foldl' f z m
    _ -> z
{-# INLINE foldl' #-}

-- | \(O(n)\). Fold the keys and values in the map using the given right-associative
-- binary operator, such that
-- @'foldrWithKey' f z == 'Prelude.foldr' ('uncurry' f) z . 'toAscList'@.
--
-- For example,
--
-- > keys map = foldrWithKey (\k x ks -> k:ks) [] map
--
-- > let f k a result = result ++ "(" ++ (show k) ++ ":" ++ a ++ ")"
-- > foldrWithKey f "Map: " (fromList [(5,"a"), (3,"b")]) == "Map: (5:a)(3:b)"
foldrWithKey :: (Key -> a -> b -> b) -> b -> IntMap a -> b
foldrWithKey f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    NonEmpty m -> NE.foldrWithKey f z m
    _ -> z
{-# INLINE foldrWithKey #-}

-- | \(O(n)\). A strict version of 'foldrWithKey'. Each application of the operator is
-- evaluated before using the result in the next application. This
-- function is strict in the starting value.
foldrWithKey' :: (Key -> a -> b -> b) -> b -> IntMap a -> b
foldrWithKey' f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    NonEmpty m -> NE.foldrWithKey' f z m
    _ -> z
{-# INLINE foldrWithKey' #-}

-- | \(O(n)\). Fold the keys and values in the map using the given left-associative
-- binary operator, such that
-- @'foldlWithKey' f z == 'Prelude.foldl' (\\z' (kx, x) -> f z' kx x) z . 'toAscList'@.
--
-- For example,
--
-- > keys = reverse . foldlWithKey (\ks k x -> k:ks) []
--
-- > let f result k a = result ++ "(" ++ (show k) ++ ":" ++ a ++ ")"
-- > foldlWithKey f "Map: " (fromList [(5,"a"), (3,"b")]) == "Map: (3:b)(5:a)"
foldlWithKey :: (a -> Key -> b -> a) -> a -> IntMap b -> a
foldlWithKey f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    NonEmpty m -> NE.foldlWithKey f z m
    _ -> z
{-# INLINE foldlWithKey #-}

-- | \(O(n)\). A strict version of 'foldlWithKey'. Each application of the operator is
-- evaluated before using the result in the next application. This
-- function is strict in the starting value.
foldlWithKey' :: (a -> Key -> b -> a) -> a -> IntMap b -> a
foldlWithKey' f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    NonEmpty m -> NE.foldlWithKey' f z m
    _ -> z
{-# INLINE foldlWithKey' #-}

-- | \(O(n)\). Fold the keys and values in the map using the given monoid, such that
--
-- @'foldMapWithKey' f = 'Prelude.fold' . 'mapWithKey' f@
--
-- This can be an asymptotically faster than 'foldrWithKey' or 'foldlWithKey' for some monoids.
--
-- @since 0.5.4
foldMapWithKey :: Monoid m => (Key -> a -> m) -> IntMap a -> m
foldMapWithKey f = go
  where
    go (NonEmpty m) = NE.foldMapWithKey f m
    go _            = mempty
{-# INLINE foldMapWithKey #-}

{--------------------------------------------------------------------
  List variations
--------------------------------------------------------------------}
-- | \(O(n)\).
-- Return all elements of the map in the ascending order of their keys.
-- Subject to list fusion.
--
-- > elems (fromList [(5,"a"), (3,"b")]) == ["b","a"]
-- > elems empty == []

elems :: IntMap a -> [a]
elems = foldr (:) []

-- | \(O(n)\). Return all keys of the map in ascending order. Subject to list
-- fusion.
--
-- > keys (fromList [(5,"a"), (3,"b")]) == [3,5]
-- > keys empty == []

keys  :: IntMap a -> [Key]
keys = foldrWithKey (\k _ ks -> k : ks) []

-- | \(O(n)\). An alias for 'toAscList'. Returns all key\/value pairs in the
-- map in ascending key order. Subject to list fusion.
--
-- > assocs (fromList [(5,"a"), (3,"b")]) == [(3,"b"), (5,"a")]
-- > assocs empty == []

assocs :: IntMap a -> [(Key,a)]
assocs = toAscList

-- | \(O(n \min(n,W))\). The set of all keys of the map.
--
-- > keysSet (fromList [(5,"a"), (3,"b")]) == Data.IntSet.fromList [3,5]
-- > keysSet empty == Data.IntSet.empty

keysSet :: IntMap a -> IntSet.IntSet
keysSet (NonEmpty m) = NE.keysSet m
keysSet Nil = IntSet.Nil

-- | \(O(n)\). Build a map from a set of keys and a function which for each key
-- computes its value.
--
-- > fromSet (\k -> replicate k 'a') (Data.IntSet.fromList [3, 5]) == fromList [(5,"aaaaa"), (3,"aaa")]
-- > fromSet undefined Data.IntSet.empty == empty

fromSet :: (Key -> a) -> IntSet.IntSet -> IntMap a
fromSet f (IntSet.Bin p m l r)
  | NonEmpty l' <- fromSet f l -- these are never empty
  , NonEmpty r' <- fromSet f r --
  = NonEmpty (NE.Bin p m l' r')
fromSet f (IntSet.Tip kx bm) = NonEmpty $ buildTree f kx bm (IntSet.suffixBitMask + 1)
  where
    -- This is slightly complicated, as we to convert the dense
    -- representation of IntSet into tree representation of IntMap.
    --
    -- We are given a nonzero bit mask 'bmask' of 'bits' bits with
    -- prefix 'prefix'. We split bmask into halves corresponding
    -- to left and right subtree. If they are both nonempty, we
    -- create a Bin node, otherwise exactly one of them is nonempty
    -- and we construct the IntMap from that half.
    buildTree g !prefix !bmask bits = case bits of
      0 -> NE.Tip prefix (g prefix)
      _ -> case intFromNat ((natFromInt bits) `shiftRL` 1) of
        bits2
          | bmask .&. ((1 `shiftLL` bits2) - 1) == 0 ->
              buildTree g (prefix + bits2) (bmask `shiftRL` bits2) bits2
          | (bmask `shiftRL` bits2) .&. ((1 `shiftLL` bits2) - 1) == 0 ->
              buildTree g prefix bmask bits2
          | otherwise ->
              NE.Bin prefix bits2
                (buildTree g prefix bmask bits2)
                (buildTree g (prefix + bits2) (bmask `shiftRL` bits2) bits2)
fromSet _ _ = Nil

{--------------------------------------------------------------------
  Lists
--------------------------------------------------------------------}

#ifdef __GLASGOW_HASKELL__
-- | @since 0.5.6.2
instance GHCExts.IsList (IntMap a) where
  type Item (IntMap a) = (Key,a)
  fromList = fromList
  toList   = toList
#endif

-- | \(O(n)\). Convert the map to a list of key\/value pairs. Subject to list
-- fusion.
--
-- > toList (fromList [(5,"a"), (3,"b")]) == [(3,"b"), (5,"a")]
-- > toList empty == []

toList :: IntMap a -> [(Key,a)]
toList = toAscList

-- | \(O(n)\). Convert the map to a list of key\/value pairs where the
-- keys are in ascending order. Subject to list fusion.
--
-- > toAscList (fromList [(5,"a"), (3,"b")]) == [(3,"b"), (5,"a")]

toAscList :: IntMap a -> [(Key,a)]
toAscList = foldrWithKey (\k x xs -> (k,x):xs) []

-- | \(O(n)\). Convert the map to a list of key\/value pairs where the keys
-- are in descending order. Subject to list fusion.
--
-- > toDescList (fromList [(5,"a"), (3,"b")]) == [(5,"a"), (3,"b")]

toDescList :: IntMap a -> [(Key,a)]
toDescList = foldlWithKey (\xs k x -> (k,x):xs) []

-- List fusion for the list generating functions.
#if __GLASGOW_HASKELL__
-- The foldrFB and foldlFB are fold{r,l}WithKey equivalents, used for list fusion.
-- They are important to convert unfused methods back, see mapFB in prelude.
foldrFB :: (Key -> a -> b -> b) -> b -> IntMap a -> b
foldrFB = foldrWithKey
{-# INLINE[0] foldrFB #-}
foldlFB :: (a -> Key -> b -> a) -> a -> IntMap b -> a
foldlFB = foldlWithKey
{-# INLINE[0] foldlFB #-}

-- Inline assocs and toList, so that we need to fuse only toAscList.
{-# INLINE assocs #-}
{-# INLINE toList #-}

-- The fusion is enabled up to phase 2 included. If it does not succeed,
-- convert in phase 1 the expanded elems,keys,to{Asc,Desc}List calls back to
-- elems,keys,to{Asc,Desc}List.  In phase 0, we inline fold{lr}FB (which were
-- used in a list fusion, otherwise it would go away in phase 1), and let compiler
-- do whatever it wants with elems,keys,to{Asc,Desc}List -- it was forbidden to
-- inline it before phase 0, otherwise the fusion rules would not fire at all.
{-# NOINLINE[0] elems #-}
{-# NOINLINE[0] keys #-}
{-# NOINLINE[0] toAscList #-}
{-# NOINLINE[0] toDescList #-}
{-# RULES "IntMap.elems" [~1] forall m . elems m = build (\c n -> foldrFB (\_ x xs -> c x xs) n m) #-}
{-# RULES "IntMap.elemsBack" [1] foldrFB (\_ x xs -> x : xs) [] = elems #-}
{-# RULES "IntMap.keys" [~1] forall m . keys m = build (\c n -> foldrFB (\k _ xs -> c k xs) n m) #-}
{-# RULES "IntMap.keysBack" [1] foldrFB (\k _ xs -> k : xs) [] = keys #-}
{-# RULES "IntMap.toAscList" [~1] forall m . toAscList m = build (\c n -> foldrFB (\k x xs -> c (k,x) xs) n m) #-}
{-# RULES "IntMap.toAscListBack" [1] foldrFB (\k x xs -> (k, x) : xs) [] = toAscList #-}
{-# RULES "IntMap.toDescList" [~1] forall m . toDescList m = build (\c n -> foldlFB (\xs k x -> c (k,x) xs) n m) #-}
{-# RULES "IntMap.toDescListBack" [1] foldlFB (\xs k x -> (k, x) : xs) [] = toDescList #-}
#endif


-- | \(O(n \min(n,W))\). Create a map from a list of key\/value pairs.
--
-- > fromList [] == empty
-- > fromList [(5,"a"), (3,"b"), (5, "c")] == fromList [(5,"c"), (3,"b")]
-- > fromList [(5,"c"), (3,"b"), (5, "a")] == fromList [(5,"a"), (3,"b")]

fromList :: [(Key,a)] -> IntMap a
fromList xs
  = Foldable.foldl' ins empty xs
  where
    ins t (k,x)  = insert k x t

-- | \(O(n \min(n,W))\). Create a map from a list of key\/value pairs with a combining function. See also 'fromAscListWith'.
--
-- > fromListWith (++) [(5,"a"), (5,"b"), (3,"b"), (3,"a"), (5,"c")] == fromList [(3, "ab"), (5, "cba")]
-- > fromListWith (++) [] == empty

fromListWith :: (a -> a -> a) -> [(Key,a)] -> IntMap a
fromListWith f xs
  = fromListWithKey (\_ x y -> f x y) xs

-- | \(O(n \min(n,W))\). Build a map from a list of key\/value pairs with a combining function. See also fromAscListWithKey'.
--
-- > let f key new_value old_value = show key ++ ":" ++ new_value ++ "|" ++ old_value
-- > fromListWithKey f [(5,"a"), (5,"b"), (3,"b"), (3,"a"), (5,"c")] == fromList [(3, "3:a|b"), (5, "5:c|5:b|a")]
-- > fromListWithKey f [] == empty

fromListWithKey :: (Key -> a -> a -> a) -> [(Key,a)] -> IntMap a
fromListWithKey f xs
  = Foldable.foldl' ins empty xs
  where
    ins t (k,x) = insertWithKey f k x t

-- | \(O(n)\). Build a map from a list of key\/value pairs where
-- the keys are in ascending order.
--
-- > fromAscList [(3,"b"), (5,"a")]          == fromList [(3, "b"), (5, "a")]
-- > fromAscList [(3,"b"), (5,"a"), (5,"b")] == fromList [(3, "b"), (5, "b")]

fromAscList :: [(Key,a)] -> IntMap a
fromAscList = fromMonoListWithKey Nondistinct (\_ x _ -> x)
{-# NOINLINE fromAscList #-}

-- | \(O(n)\). Build a map from a list of key\/value pairs where
-- the keys are in ascending order, with a combining function on equal keys.
-- /The precondition (input list is ascending) is not checked./
--
-- > fromAscListWith (++) [(3,"b"), (5,"a"), (5,"b")] == fromList [(3, "b"), (5, "ba")]

fromAscListWith :: (a -> a -> a) -> [(Key,a)] -> IntMap a
fromAscListWith f = fromMonoListWithKey Nondistinct (\_ x y -> f x y)
{-# NOINLINE fromAscListWith #-}

-- | \(O(n)\). Build a map from a list of key\/value pairs where
-- the keys are in ascending order, with a combining function on equal keys.
-- /The precondition (input list is ascending) is not checked./
--
-- > let f key new_value old_value = (show key) ++ ":" ++ new_value ++ "|" ++ old_value
-- > fromAscListWithKey f [(3,"b"), (5,"a"), (5,"b")] == fromList [(3, "b"), (5, "5:b|a")]

fromAscListWithKey :: (Key -> a -> a -> a) -> [(Key,a)] -> IntMap a
fromAscListWithKey f = fromMonoListWithKey Nondistinct f
{-# NOINLINE fromAscListWithKey #-}

-- | \(O(n)\). Build a map from a list of key\/value pairs where
-- the keys are in ascending order and all distinct.
-- /The precondition (input list is strictly ascending) is not checked./
--
-- > fromDistinctAscList [(3,"b"), (5,"a")] == fromList [(3, "b"), (5, "a")]

fromDistinctAscList :: [(Key,a)] -> IntMap a
fromDistinctAscList = fromMonoListWithKey Distinct (\_ x _ -> x)
{-# NOINLINE fromDistinctAscList #-}

-- | \(O(n)\). Build a map from a list of key\/value pairs with monotonic keys
-- and a combining function.
--
-- The precise conditions under which this function works are subtle:
-- For any branch mask, keys with the same prefix w.r.t. the branch
-- mask must occur consecutively in the list.

fromMonoListWithKey :: Distinct -> (Key -> a -> a -> a) -> [(Key,a)] -> IntMap a
fromMonoListWithKey distinct f =
  \ t ->
    case t of
      [] -> Nil
      ts -> NonEmpty $ NE.fromMonoListWithKey distinct f ts
{-# INLINE fromMonoListWithKey #-}

{--------------------------------------------------------------------
  Eq
--------------------------------------------------------------------}
instance Eq a => Eq (IntMap a) where
  t1 == t2  = equal t1 t2
  t1 /= t2  = nequal t1 t2

equal :: Eq a => IntMap a -> IntMap a -> Bool
equal (NonEmpty m1) (NonEmpty m2) = m1 == m2
equal Nil Nil = True
equal _   _   = False

nequal :: Eq a => IntMap a -> IntMap a -> Bool
nequal (NonEmpty m1) (NonEmpty m2) = m1 /= m2
nequal Nil Nil = False
nequal _   _   = True

-- | @since 0.5.9
instance Eq1 IntMap where
  liftEq eq (NonEmpty m1) (NonEmpty m2) = liftEq eq m1 m2
  liftEq _eq Nil Nil = True
  liftEq _eq _   _   = False

{--------------------------------------------------------------------
  Ord
--------------------------------------------------------------------}

instance Ord a => Ord (IntMap a) where
    compare m1 m2 = compare (toList m1) (toList m2)

-- | @since 0.5.9
instance Ord1 IntMap where
  liftCompare cmp m n =
    liftCompare (liftCompare cmp) (toList m) (toList n)

{--------------------------------------------------------------------
  Functor
--------------------------------------------------------------------}

instance Functor IntMap where
    fmap = map

#ifdef __GLASGOW_HASKELL__
    a <$ NonEmpty m  = NonEmpty (a <$ m)
    _ <$ Nil         = Nil
#endif

{--------------------------------------------------------------------
  Show
--------------------------------------------------------------------}

instance Show a => Show (IntMap a) where
  showsPrec d m   = showParen (d > 10) $
    showString "fromList " . shows (toList m)

-- | @since 0.5.9
instance Show1 IntMap where
    liftShowsPrec sp sl d m =
        showsUnaryWith (liftShowsPrec sp' sl') "fromList" d (toList m)
      where
        sp' = liftShowsPrec sp sl
        sl' = liftShowList sp sl

{--------------------------------------------------------------------
  Read
--------------------------------------------------------------------}
instance (Read e) => Read (IntMap e) where
#ifdef __GLASGOW_HASKELL__
  readPrec = parens $ prec 10 $ do
    Ident "fromList" <- lexP
    xs <- readPrec
    return (fromList xs)

  readListPrec = readListPrecDefault
#else
  readsPrec p = readParen (p > 10) $ \ r -> do
    ("fromList",s) <- lex r
    (xs,t) <- reads s
    return (fromList xs,t)
#endif

-- | @since 0.5.9
instance Read1 IntMap where
    liftReadsPrec rp rl = readsData $
        readsUnaryWith (liftReadsPrec rp' rl') "fromList" fromList
      where
        rp' = liftReadsPrec rp rl
        rl' = liftReadList rp rl

{--------------------------------------------------------------------
  Helpers
--------------------------------------------------------------------}

{--------------------------------------------------------------------
  @bin@ assures that we never have empty trees within a tree.
--------------------------------------------------------------------}
bin :: Prefix -> Mask -> IntMap a -> IntMap a -> IntMap a
bin p m (NonEmpty l) (NonEmpty r) = NonEmpty (NE.Bin p m l r)
bin _ _ l Nil = l
bin _ _ Nil r = r
{-# INLINE bin #-}

-- binCheckLeft only checks that the left subtree is non-empty
binCheckLeft :: Prefix -> Mask -> IntMap a -> NE.IntMap a -> IntMap a
binCheckLeft p m (NonEmpty l) r = NonEmpty $ NE.Bin p m l r
binCheckLeft _ _ Nil r = NonEmpty $ r
{-# INLINE binCheckLeft #-}

-- binCheckRight only checks that the right subtree is non-empty
binCheckRight :: Prefix -> Mask -> NE.IntMap a -> IntMap a -> IntMap a
binCheckRight p m l (NonEmpty r) = NonEmpty $ NE.Bin p m l r
binCheckRight _ _ l Nil = NonEmpty $ l
{-# INLINE binCheckRight #-}

{--------------------------------------------------------------------
  Endian independent bit twiddling
--------------------------------------------------------------------}

-- | Should this key follow the left subtree of a 'Bin' with switching
-- bit @m@? N.B., the answer is only valid when @match i p m@ is true.
zero :: Key -> Mask -> Bool
zero i m
  = (natFromInt i) .&. (natFromInt m) == 0
{-# INLINE zero #-}

nomatch,match :: Key -> Prefix -> Mask -> Bool

-- | Does the key @i@ differ from the prefix @p@ before getting to
-- the switching bit @m@?
nomatch i p m
  = (mask i m) /= p
{-# INLINE nomatch #-}

-- | Does the key @i@ match the prefix @p@ (up to but not including
-- bit @m@)?
match i p m
  = (mask i m) == p
{-# INLINE match #-}


-- | The prefix of key @i@ up to (but not including) the switching
-- bit @m@.
mask :: Key -> Mask -> Prefix
mask i m
  = maskW (natFromInt i) (natFromInt m)
{-# INLINE mask #-}


{--------------------------------------------------------------------
  Big endian operations
--------------------------------------------------------------------}

-- | The prefix of key @i@ up to (but not including) the switching
-- bit @m@.
maskW :: Nat -> Nat -> Prefix
maskW i m
  = intFromNat (i .&. ((-m) `xor` m))
{-# INLINE maskW #-}

-- | Does the left switching bit specify a shorter prefix?
shorter :: Mask -> Mask -> Bool
shorter m1 m2
  = (natFromInt m1) > (natFromInt m2)
{-# INLINE shorter #-}

-- | The first switching bit where the two prefixes disagree.
branchMask :: Prefix -> Prefix -> Mask
branchMask p1 p2
  = intFromNat (highestBitMask (natFromInt p1 `xor` natFromInt p2))
{-# INLINE branchMask #-}

{--------------------------------------------------------------------
  Utilities
--------------------------------------------------------------------}

-- | \(O(1)\).  Decompose a map into pieces based on the structure
-- of the underlying tree. This function is useful for consuming a
-- map in parallel.
--
-- No guarantee is made as to the sizes of the pieces; an internal, but
-- deterministic process determines this.  However, it is guaranteed that the
-- pieces returned will be in ascending order (all elements in the first submap
-- less than all elements in the second, and so on).
--
-- Examples:
--
-- > splitRoot (fromList (zip [1..6::Int] ['a'..])) ==
-- >   [fromList [(1,'a'),(2,'b'),(3,'c')],fromList [(4,'d'),(5,'e'),(6,'f')]]
--
-- > splitRoot empty == []
--
--  Note that the current implementation does not return more than two submaps,
--  but you should not depend on this behaviour because it can change in the
--  future without notice.
splitRoot :: IntMap a -> [IntMap a]
splitRoot orig =
  case orig of
    Nil -> []
    NonEmpty m -> fmap NonEmpty $ NE.splitRoot m
{-# INLINE splitRoot #-}


{--------------------------------------------------------------------
  Debugging
--------------------------------------------------------------------}

-- | \(O(n \min(n,W))\). Show the tree that implements the map. The tree is shown
-- in a compressed, hanging format.
showTree :: Show a => IntMap a -> String
showTree s
  = showTreeWith True False s


{- | \(O(n \min(n,W))\). The expression (@'showTreeWith' hang wide map@) shows
 the tree that implements the map. If @hang@ is
 'True', a /hanging/ tree is shown otherwise a rotated tree is shown. If
 @wide@ is 'True', an extra wide version is shown.
-}
showTreeWith :: Show a => Bool -> Bool -> IntMap a -> String
showTreeWith hang wide t
  | hang      = (showsTreeHang wide [] t) ""
  | otherwise = (showsTree wide [] [] t) ""

showsTree :: Show a => Bool -> [String] -> [String] -> IntMap a -> ShowS
showsTree wide lbars rbars t = case t of
  NonEmpty m -> NE.showsTree wide lbars rbars m
  Nil -> showsBars lbars . showString "|\n"

showsTreeHang :: Show a => Bool -> [String] -> IntMap a -> ShowS
showsTreeHang wide bars t = case t of
  NonEmpty m -> NE.showsTreeHang wide bars m
  Nil -> showsBars bars . showString "|\n"

showsBars :: [String] -> ShowS
showsBars bars
  = case bars of
      [] -> id
      _ : tl -> showString (concat (reverse tl)) . showString node

node :: String
node = "+--"
