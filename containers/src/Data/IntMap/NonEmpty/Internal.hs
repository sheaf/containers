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
-- Module      :  Data.IntMap.NonEmpty.Internal
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

module Data.IntMap.NonEmpty.Internal (
    -- * Non-empty map type
      IntMap(..), Key          -- instance Eq,Show

    -- * Operators
    , (!), (!?)

    -- * Query
    , size
    , member
    , notMember
    , lookup
    , find
    , findWithDefault
    , lookupLT
    , lookupGT
    , lookupLE
    , lookupGE
    , disjoint

    -- * Construction
    , singleton

    -- ** Insertion
    , insert
    , insertWith
    , insertWithKey
    , insertLookupWithKey

    -- ** Update
    , adjust
    , adjustWithKey

    -- * Combine

    -- ** Union
    , union
    , unionWith
    , unionWithKey

    -- ** General combining function
    , SimpleWhenMissing
    , SimpleWhenMatched
    , runWhenMatched
    , runWhenMissing
    , merge
    -- *** @WhenMatched@ tactics
    , zipWithMatched
    -- *** @WhenMissing@ tactics
    , preserveMissing
    , mapMissing

    -- ** Applicative general combining function
    , WhenMissing (..)
    , WhenMatched (..)
    , mergeA
    -- *** @WhenMatched@ tactics
    -- | The tactics described for 'merge' work for
    -- 'mergeA' as well. Furthermore, the following
    -- are available.
    , zipWithAMatched
    -- *** @WhenMissing@ tactics
    -- | The tactics described for 'merge' work for
    -- 'mergeA' as well. Furthermore, the following
    -- are available.
    , traverseMissing

    -- ** Deprecated general combining function
    , mergeWithKey

    -- * Traversal
    -- ** Map
    , map
    , mapWithKey
    , traverseWithKey
    , mapAccum
    , mapAccumL
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

    , Distinct(..)
    , fromMonoListWithKey

    -- * Utilities
    , splitRoot

    -- * Submap
    , isSubmapOf, isSubmapOfBy
    , isProperSubmapOf, isProperSubmapOfBy

    -- * Min\/Max
    , findMin
    , findMax
    , updateMinWithKey
    , updateMaxWithKey

    -- * Debugging
    , showTree
    , showTreeWith

    -- * Internal types
    , Mask, Prefix, Nat

    -- * Utility
    , natFromInt
    , intFromNat
    , link
    , linkWithMask
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

    -- * Show functions
    , showsTree, showsTreeHang
    ) where

import Data.Functor.Identity (Identity (..))
import Data.Semigroup (Semigroup(stimes))
#if !(MIN_VERSION_base(4,11,0))
import Data.Semigroup (Semigroup((<>)))
#endif
import Data.Semigroup (stimesIdempotent)
import Data.Functor.Classes

import Control.DeepSeq (NFData(rnf))
import Data.Bits
import qualified Data.Foldable as Foldable
import Utils.Containers.Internal.Prelude hiding
  (lookup, map, filter, foldr, foldl, null)
import Prelude ()

import Data.IntSet.Internal (Key)
import qualified Data.IntSet.Internal as IntSet
import Utils.Containers.Internal.BitUtil

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


-- | A non-empty map of integers to values @a@.

-- See Note: Order of constructors
data IntMap a
  = Bin {-# UNPACK #-} !Prefix
        {-# UNPACK #-} !Mask
        !(IntMap a)
        !(IntMap a)
-- Fields:
--   prefix: The most significant bits shared by all keys in this Bin.
--   mask: The switching bit to determine if a key should follow the left
--         or right subtree of a 'Bin'.
-- Invariant: The Mask is a power of 2. It is the largest bit position at which
--            two keys of the map differ.
-- Invariant: Prefix is the common high-order bits that all elements share to
--            the left of the Mask bit.
-- Invariant: In (Bin prefix mask left right), left consists of the elements that
--            don't have the mask bit set; right is all the elements that do.
  | Tip {-# UNPACK #-} !Key a

type Prefix = Int
type Mask   = Int

-- | @since 0.6.6
deriving instance Lift a => Lift (IntMap a)

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

infixl 9 !?

{--------------------------------------------------------------------
  Types
--------------------------------------------------------------------}

-- | @since 0.5.7
instance Semigroup (IntMap a) where
    (<>)    = union
    stimes  = stimesIdempotent

-- | Folds in order of increasing key.
instance Foldable.Foldable IntMap where
  fold = go
    where go (Tip _ v) = v
          go (Bin _ m l r)
            | m < 0     = go r `mappend` go l
            | otherwise = go l `mappend` go r
  {-# INLINABLE fold #-}
  foldr = foldr
  {-# INLINE foldr #-}
  foldl = foldl
  {-# INLINE foldl #-}
  foldMap f t = go t
    where go (Tip _ v) = f v
          go (Bin _ m l r)
            | m < 0     = go r `mappend` go l
            | otherwise = go l `mappend` go r
  {-# INLINE foldMap #-}
  foldl' = foldl'
  {-# INLINE foldl' #-}
  foldr' = foldr'
  {-# INLINE foldr' #-}
  length = size
  {-# INLINE length #-}
  null   = const False
  {-# INLINE null #-}
  toList = elems -- NB: Foldable.toList /= IntMap.toList
  {-# INLINE toList #-}
  elem = go
    where go x (Tip _ y) = x == y
          go x (Bin _ _ l r) = go x l || go x r
  {-# INLINABLE elem #-}
  maximum = start
    where start (Tip _ y) = y
          start (Bin _ m l r)
            | m < 0     = go (start r) l
            | otherwise = go (start l) r

          go m (Tip _ y) = max m y
          go m (Bin _ _ l r) = go (go m l) r
  {-# INLINABLE maximum #-}
  minimum = start
    where start (Tip _ y) = y
          start (Bin _ m l r)
            | m < 0     = go (start r) l
            | otherwise = go (start l) r

          go m (Tip _ y) = min m y
          go m (Bin _ _ l r) = go (go m l) r
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
    rnf (Tip _ v) = rnf v
    rnf (Bin _ _ l r) = rnf l `seq` rnf r

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

-- | \(O(n)\). Number of elements in the map.
--
-- > size (singleton 1 'a')                       == 1
-- > size (fromList([(1,'a'), (2,'c'), (3,'b')])) == 3
size :: IntMap a -> Int
size = go 0
  where
    go !acc (Bin _ _ l r) = go (go acc l) r
    go acc (Tip _ _) = 1 + acc

-- | \(O(\min(n,W))\). Is the key a member of the map?
--
-- > member 5 (fromList [(5,'a'), (3,'b')]) == True
-- > member 1 (fromList [(5,'a'), (3,'b')]) == False

-- See Note: Local 'go' functions and capturing]
member :: Key -> IntMap a -> Bool
member !k = go
  where
    go (Bin p m l r) | nomatch k p m = False
                     | zero k m  = go l
                     | otherwise = go r
    go (Tip kx _) = k == kx

-- | \(O(\min(n,W))\). Is the key not a member of the map?
--
-- > notMember 5 (fromList [(5,'a'), (3,'b')]) == False
-- > notMember 1 (fromList [(5,'a'), (3,'b')]) == True

notMember :: Key -> IntMap a -> Bool
notMember k m = not $ member k m

-- | \(O(\min(n,W))\). Lookup the value at a key in the map. See also 'Data.Map.lookup'.

-- See Note: Local 'go' functions and capturing
lookup :: Key -> IntMap a -> Maybe a
lookup !k = go
  where
    go (Bin _p m l r) | zero k m  = go l
                      | otherwise = go r
    go (Tip kx x) | k == kx   = Just x
                  | otherwise = Nothing

-- See Note: Local 'go' functions and capturing]
find :: Key -> IntMap a -> a
find !k = go
  where
    go (Bin _p m l r) | zero k m  = go l
                      | otherwise = go r
    go (Tip kx x) | k == kx   = x
                  | otherwise = not_found

    not_found = error ("IntMap.!: key " ++ show k ++ " is not an element of the map")

-- | \(O(\min(n,W))\). The expression @('findWithDefault' def k map)@
-- returns the value at key @k@ or returns @def@ when the key is not an
-- element of the map.
--
-- > findWithDefault 'x' 1 (fromList [(5,'a'), (3,'b')]) == 'x'
-- > findWithDefault 'x' 5 (fromList [(5,'a'), (3,'b')]) == 'a'

-- See Note: Local 'go' functions and capturing]
findWithDefault :: a -> Key -> IntMap a -> a
findWithDefault def !k = go
  where
    go (Bin p m l r) | nomatch k p m = def
                     | zero k m  = go l
                     | otherwise = go r
    go (Tip kx x) | k == kx   = x
                  | otherwise = def

-- | \(O(\min(n,W))\). Find largest key smaller than the given one and return the
-- corresponding (key, value) pair.
--
-- > lookupLT 3 (fromList [(3,'a'), (5,'b')]) == Nothing
-- > lookupLT 4 (fromList [(3,'a'), (5,'b')]) == Just (3, 'a')

-- See Note: Local 'go' functions and capturing.
lookupLT :: Key -> IntMap a -> Maybe (Key, a)
lookupLT !k t = case t of
    Bin _ m l r | m < 0 -> if k >= 0 then go r l else go2 r
    _ -> go2 t
  where
    go def (Bin p m l r)
      | nomatch k p m = if k < p then unsafeFindMax def else unsafeFindMax r
      | zero k m  = go def l
      | otherwise = go l r
    go def (Tip ky y)
      | k <= ky   = unsafeFindMax def
      | otherwise = Just (ky, y)
    go2 (Bin p m l r)
      | nomatch k p m = if k < p then Nothing else unsafeFindMax r
      | zero k m  = go2 l
      | otherwise = go l r
    go2 (Tip ky y)
      | k <= ky   = Nothing
      | otherwise = Just (ky, y)

-- | \(O(\min(n,W))\). Find smallest key greater than the given one and return the
-- corresponding (key, value) pair.
--
-- > lookupGT 4 (fromList [(3,'a'), (5,'b')]) == Just (5, 'b')
-- > lookupGT 5 (fromList [(3,'a'), (5,'b')]) == Nothing

-- See Note: Local 'go' functions and capturing.
lookupGT :: Key -> IntMap a -> Maybe (Key, a)
lookupGT !k t = case t of
    Bin _ m l r | m < 0 -> if k >= 0 then go2 l else go l r
    _ -> go2 t
  where
    go def (Bin p m l r)
      | nomatch k p m = if k < p then unsafeFindMin l else unsafeFindMin def
      | zero k m  = go r l
      | otherwise = go def r
    go def (Tip ky y)
      | k >= ky   = unsafeFindMin def
      | otherwise = Just (ky, y)
    go2 (Bin p m l r)
      | nomatch k p m = if k < p then unsafeFindMin l else Nothing
      | zero k m  = go r l
      | otherwise = go2 r
    go2 (Tip ky y)
      | k >= ky   = Nothing
      | otherwise = Just (ky, y)

-- | \(O(\min(n,W))\). Find largest key smaller or equal to the given one and return
-- the corresponding (key, value) pair.
--
-- > lookupLE 2 (fromList [(3,'a'), (5,'b')]) == Nothing
-- > lookupLE 4 (fromList [(3,'a'), (5,'b')]) == Just (3, 'a')
-- > lookupLE 5 (fromList [(3,'a'), (5,'b')]) == Just (5, 'b')

-- See Note: Local 'go' functions and capturing.
lookupLE :: Key -> IntMap a -> Maybe (Key, a)
lookupLE !k t = case t of
    Bin _ m l r | m < 0 -> if k >= 0 then go r l else go2 r
    _ -> go2 t
  where
    go def (Bin p m l r)
      | nomatch k p m = if k < p then unsafeFindMax def else unsafeFindMax r
      | zero k m  = go def l
      | otherwise = go l r
    go def (Tip ky y)
      | k < ky    = unsafeFindMax def
      | otherwise = Just (ky, y)
    go2 (Bin p m l r)
      | nomatch k p m = if k < p then Nothing else unsafeFindMax r
      | zero k m  = go2 l
      | otherwise = go l r
    go2 (Tip ky y)
      | k < ky    = Nothing
      | otherwise = Just (ky, y)

-- | \(O(\min(n,W))\). Find smallest key greater or equal to the given one and return
-- the corresponding (key, value) pair.
--
-- > lookupGE 3 (fromList [(3,'a'), (5,'b')]) == Just (3, 'a')
-- > lookupGE 4 (fromList [(3,'a'), (5,'b')]) == Just (5, 'b')
-- > lookupGE 6 (fromList [(3,'a'), (5,'b')]) == Nothing

-- See Note: Local 'go' functions and capturing.
lookupGE :: Key -> IntMap a -> Maybe (Key, a)
lookupGE !k t = case t of
    Bin _ m l r | m < 0 -> if k >= 0 then go2 l else go l r
    _ -> go2 t
  where
    go def (Bin p m l r)
      | nomatch k p m = if k < p then unsafeFindMin l else unsafeFindMin def
      | zero k m  = go r l
      | otherwise = go def r
    go def (Tip ky y)
      | k > ky    = unsafeFindMin def
      | otherwise = Just (ky, y)
    go2 (Bin p m l r)
      | nomatch k p m = if k < p then unsafeFindMin l else Nothing
      | zero k m  = go r l
      | otherwise = go2 r
    go2 (Tip ky y)
      | k > ky    = Nothing
      | otherwise = Just (ky, y)

-- Helper function for lookupGE and lookupGT. It assumes that if a Bin node is
-- given, it has m > 0.
unsafeFindMin :: IntMap a -> Maybe (Key, a)
unsafeFindMin (Tip ky y) = Just (ky, y)
unsafeFindMin (Bin _ _ l _) = unsafeFindMin l

-- Helper function for lookupLE and lookupLT. It assumes that if a Bin node is
-- given, it has m > 0.
unsafeFindMax :: IntMap a -> Maybe (Key, a)
unsafeFindMax (Tip ky y) = Just (ky, y)
unsafeFindMax (Bin _ _ _ r) = unsafeFindMax r

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
disjoint (Tip kx _) ys = notMember kx ys
disjoint xs (Tip ky _) = notMember ky xs
disjoint t1@(Bin p1 m1 l1 r1) t2@(Bin p2 m2 l2 r2)
  | shorter m1 m2 = disjoint1
  | shorter m2 m1 = disjoint2
  | p1 == p2      = disjoint l1 l2 && disjoint r1 r2
  | otherwise     = True
  where
    disjoint1 | nomatch p2 p1 m1 = True
              | zero p2 m1       = disjoint l1 t2
              | otherwise        = disjoint r1 t2
    disjoint2 | nomatch p1 p2 m2 = True
              | zero p1 m2       = disjoint t1 l2
              | otherwise        = disjoint t1 r2

{--------------------------------------------------------------------
  Construction
--------------------------------------------------------------------}

-- | \(O(1)\). A map of one element.
--
-- > singleton 1 'a'        == fromList [(1, 'a')]
-- > size (singleton 1 'a') == 1

singleton :: Key -> a -> IntMap a
singleton k x
  = Tip k x
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
insert !k x t@(Bin p m l r)
  | nomatch k p m = link k (Tip k x) p t
  | zero k m      = Bin p m (insert k x l) r
  | otherwise     = Bin p m l (insert k x r)
insert k x t@(Tip ky _)
  | k==ky         = Tip k x
  | otherwise     = link k (Tip k x) ky t

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
insertWithKey f !k x t@(Bin p m l r)
  | nomatch k p m = link k (Tip k x) p t
  | zero k m      = Bin p m (insertWithKey f k x l) r
  | otherwise     = Bin p m l (insertWithKey f k x r)
insertWithKey f k x t@(Tip ky y)
  | k == ky       = Tip k (f k x y)
  | otherwise     = link k (Tip k x) ky t

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
insertLookupWithKey f !k x t@(Bin p m l r)
  | nomatch k p m = (Nothing,link k (Tip k x) p t)
  | zero k m      = let (found,l') = insertLookupWithKey f k x l
                    in (found,Bin p m l' r)
  | otherwise     = let (found,r') = insertLookupWithKey f k x r
                    in (found,Bin p m l r')
insertLookupWithKey f k x t@(Tip ky y)
  | k == ky       = (Just y,Tip k (f k x y))
  | otherwise     = (Nothing,link k (Tip k x) ky t)


{--------------------------------------------------------------------
  Update
--------------------------------------------------------------------}

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
adjustWithKey f !k (Bin p m l r)
  | zero k m      = Bin p m (adjustWithKey f k l) r
  | otherwise     = Bin p m l (adjustWithKey f k r)
adjustWithKey f k t@(Tip ky y)
  | k == ky       = Tip ky (f k y)
  | otherwise     = t

{--------------------------------------------------------------------
  Union
--------------------------------------------------------------------}

-- | \(O(n+m)\). The (left-biased) union of two maps.
-- It prefers the first map when duplicate keys are encountered,
-- i.e. (@'union' == 'unionWith' 'const'@).
--
-- > union (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == fromList [(3, "b"), (5, "a"), (7, "C")]

union :: IntMap a -> IntMap a -> IntMap a
union m1 m2
  = mergeWithKey const id id m1 m2

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
unionWithKey f m1 m2
  = mergeWithKey (\(Tip k1 x1) (Tip _k2 x2) -> Tip k1 (f k1 x1 x2)) id id m1 m2

{--------------------------------------------------------------------
  MergeWithKey
--------------------------------------------------------------------}

-- | \(O(n+m)\). A high-performance universal combining function. Using
-- 'mergeWithKey', all combining functions can be defined without any loss of
-- efficiency.
--
-- Please make sure you know what is going on when using 'mergeWithKey',
-- otherwise you can be surprised by unexpected code growth or even
-- corruption of the data structure.
--
-- When 'mergeWithKey' is given three arguments, it is inlined to the call
-- site. You should therefore use 'mergeWithKey' only to define your custom
-- combining functions.

mergeWithKey :: (IntMap a -> IntMap b -> IntMap c) -> (IntMap a -> IntMap c) -> (IntMap b -> IntMap c)
             -> IntMap a -> IntMap b -> IntMap c
mergeWithKey f g1 g2 = go
  where
    go t1@(Bin p1 m1 l1 r1) t2@(Bin p2 m2 l2 r2)
      | shorter m1 m2  = merge1
      | shorter m2 m1  = merge2
      | p1 == p2       = Bin p1 m1 (go l1 l2) (go r1 r2)
      | otherwise      = link p1 (g1 t1) p2 (g2 t2)
      where
        merge1 | nomatch p2 p1 m1  = link p1 (g1 t1) p2 (g2 t2)
               | zero p2 m1        = Bin p1 m1 (go l1 t2) (g1 r1)
               | otherwise         = Bin p1 m1 (g1 l1) (go r1 t2)
        merge2 | nomatch p1 p2 m2  = link p1 (g1 t1) p2 (g2 t2)
               | zero p1 m2        = Bin p2 m2 (go t1 l2) (g2 r2)
               | otherwise         = Bin p2 m2 (g2 l2) (go t1 r2)

    go t1'@(Bin _ _ _ _) t2'@(Tip k2' _) = merge0 t2' k2' t1'
      where
        merge0 t2 k2 t1@(Bin p1 m1 l1 r1)
          | nomatch k2 p1 m1 = link p1 (g1 t1) k2 (g2 t2)
          | zero k2 m1 = Bin p1 m1 (merge0 t2 k2 l1) (g1 r1)
          | otherwise  = Bin p1 m1 (g1 l1) (merge0 t2 k2 r1)
        merge0 t2 k2 t1@(Tip k1 _)
          | k1 == k2 = f t1 t2
          | otherwise = link k1 (g1 t1) k2 (g2 t2)

    go t1'@(Tip k1' _) t2' = merge0 t1' k1' t2'
      where
        merge0 t1 k1 t2@(Bin p2 m2 l2 r2)
          | nomatch k1 p2 m2 = link k1 (g1 t1) p2 (g2 t2)
          | zero k1 m2 = Bin p2 m2 (merge0 t1 k1 l2) (g2 r2)
          | otherwise  = Bin p2 m2 (g2 l2) (merge0 t1 k1 r2)
        merge0 t1 k1 t2@(Tip k2 _)
          | k1 == k2 = f t1 t2
          | otherwise = link k1 (g1 t1) k2 (g2 t2)
{-# INLINE mergeWithKey #-}


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
  { missingSubtree :: IntMap x -> f (IntMap y)
  , missingKey :: Key -> x -> f y}

-- | @since 0.5.9
instance (Applicative f, Monad f) => Functor (WhenMissing f x) where
  fmap = mapWhenMissing
  {-# INLINE fmap #-}


-- | @since 0.5.9
instance (Applicative f, Monad f) => Category.Category (WhenMissing f)
  where
    id = preserveMissing
    f . g =
      traverseMissing $ \ k x -> do
        q <- missingKey g k x
        missingKey f k q
    {-# INLINE id #-}
    {-# INLINE (.) #-}


-- | Equivalent to @ReaderT k (ReaderT x (MaybeT f))@.
--
-- @since 0.5.9
instance (Applicative f, Monad f) => Applicative (WhenMissing f x) where
  pure x = mapMissing (\ _ _ -> x)
  f <*> g =
    traverseMissing $ \k x -> do
      r <- missingKey f k x
      (pure $!) . r =<< missingKey g k x
  {-# INLINE pure #-}
  {-# INLINE (<*>) #-}


-- | Equivalent to @ReaderT k (ReaderT x (MaybeT f))@.
--
-- @since 0.5.9
instance (Applicative f, Monad f) => Monad (WhenMissing f x) where
  m >>= f =
    traverseMissing $ \k x -> do
      r <- missingKey m k x
      missingKey (f r) k x
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
  , missingKey     = \k x -> missingKey t k x >>= \q -> (pure $! f q) }
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
  , missingKey     = \k x -> f <$> missingKey t k x }
{-# INLINE mapGentlyWhenMissing #-}


-- | Map covariantly over a @'WhenMatched' f k x@, using only a
-- 'Functor f' constraint.
mapGentlyWhenMatched
  :: Functor f
  => (a -> b)
  -> WhenMatched f x y a
  -> WhenMatched f x y b
mapGentlyWhenMatched f t =
  zipWithAMatched $ \k x y -> f <$> runWhenMatched t k x y
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
-- of a function of type @Key -> x -> y -> f z@.
--
-- @since 0.5.9
newtype WhenMatched f x y z = WhenMatched
  { matchedKey :: Key -> x -> y -> f z }


-- | Along with zipWithMaybeAMatched, witnesses the isomorphism
-- between @WhenMatched f x y z@ and @Key -> x -> y -> f z@.
--
-- @since 0.5.9
runWhenMatched :: WhenMatched f x y z -> Key -> x -> y -> f z
runWhenMatched = matchedKey
{-# INLINE runWhenMatched #-}


-- | Along with traverseMaybeMissing, witnesses the isomorphism
-- between @WhenMissing f x y@ and @Key -> x -> f y@.
--
-- @since 0.5.9
runWhenMissing :: WhenMissing f x y -> Key -> x -> f y
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
      zipWithAMatched $ \k x y -> do
        r <- runWhenMatched g k x y
        runWhenMatched f k x r
    {-# INLINE id #-}
    {-# INLINE (.) #-}


-- | Equivalent to @ReaderT Key (ReaderT x (ReaderT y (MaybeT f)))@
--
-- @since 0.5.9
instance (Monad f, Applicative f) => Applicative (WhenMatched f x y) where
  pure x = zipWithMatched (\_ _ _ -> x)
  fs <*> xs =
    zipWithAMatched $ \k x y -> do
      r <- runWhenMatched fs k x y
      (pure $!) . r =<< runWhenMatched xs k x y
  {-# INLINE pure #-}
  {-# INLINE (<*>) #-}


-- | Equivalent to @ReaderT Key (ReaderT x (ReaderT y (MaybeT f)))@
--
-- @since 0.5.9
instance (Monad f, Applicative f) => Monad (WhenMatched f x y) where
  m >>= f =
    zipWithAMatched $ \k x y -> do
      r <- runWhenMatched m k x y
      runWhenMatched (f r) k x y
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
  WhenMatched $ \k x y -> fmap f (g k x y)
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
zipWithMatched f = WhenMatched $ \ k x y -> pure $ f k x y
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
zipWithAMatched f = WhenMatched $ \ k x y -> f k x y
{-# INLINE zipWithAMatched #-}

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
  { missingSubtree = pure
  , missingKey     = \_ v -> pure v }
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
  { missingSubtree = \m -> pure $! mapWithKey f m
  , missingKey     = \k x -> pure $ f k x }
{-# INLINE mapMissing #-}

-- | Traverse over the entries whose keys are missing from the other
-- map.
--
-- @since 0.5.9
traverseMissing
  :: Applicative f => (Key -> x -> f y) -> WhenMissing f x y
traverseMissing f = WhenMissing
  { missingSubtree = traverseWithKey f
  , missingKey = \k x -> f k x }
{-# INLINE traverseMissing #-}

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

    -- This case is already covered below.
    -- go (Tip k1 x1) (Tip k2 x2) = mergeTips k1 x1 k2 x2

    go (Tip k1 x1) t2' = merge2 t2'
      where
        merge2 t2@(Bin p2 m2 l2 r2)
          | nomatch k1 p2 m2 = linkA k1 (subsingletonBy g1k k1 x1) p2 (g2t t2)
          | zero k1 m2       = binA p2 m2 (merge2 l2) (g2t r2)
          | otherwise        = binA p2 m2 (g2t l2) (merge2 r2)
        merge2 (Tip k2 x2)   = mergeTips k1 x1 k2 x2

    go t1' (Tip k2 x2) = merge1 t1'
      where
        merge1 t1@(Bin p1 m1 l1 r1)
          | nomatch k2 p1 m1 = linkA p1 (g1t t1) k2 (subsingletonBy g2k k2 x2)
          | zero k2 m1       = binA p1 m1 (merge1 l1) (g1t r1)
          | otherwise        = binA p1 m1 (g1t l1) (merge1 r1)
        merge1 (Tip k1 x1)   = mergeTips k1 x1 k2 x2

    go t1@(Bin p1 m1 l1 r1) t2@(Bin p2 m2 l2 r2)
      | shorter m1 m2  = merge1
      | shorter m2 m1  = merge2
      | p1 == p2       = binA p1 m1 (go l1 l2) (go r1 r2)
      | otherwise      = linkA p1 (g1t t1) p2 (g2t t2)
      where
        merge1 | nomatch p2 p1 m1  = linkA p1 (g1t t1) p2 (g2t t2)
               | zero p2 m1        = binA p1 m1 (go  l1 t2) (g1t r1)
               | otherwise         = binA p1 m1 (g1t l1)    (go  r1 t2)
        merge2 | nomatch p1 p2 m2  = linkA p1 (g1t t1) p2 (g2t t2)
               | zero p1 m2        = binA p2 m2 (go  t1 l2) (g2t    r2)
               | otherwise         = binA p2 m2 (g2t    l2) (go  t1 r2)

    subsingletonBy gk k x = Tip k <$> gk k x
    {-# INLINE subsingletonBy #-}

    mergeTips k1 x1 k2 x2
      | k1 == k2  = Tip k1 <$> f k1 x1 x2
      | k1 <  k2  = liftA2 (subdoubleton k1 k2) (g1k k1 x1) (g2k k2 x2)
        {-
        = link_ k1 k2 <$> subsingletonBy g1k k1 x1 <*> subsingletonBy g2k k2 x2
        -}
      | otherwise = liftA2 (subdoubleton k2 k1) (g2k k2 x2) (g1k k1 x1)
    {-# INLINE mergeTips #-}

    subdoubleton k1 k2 y1 y2 = link k1 (Tip k1 y1) k2 (Tip k2 y2)
    {-# INLINE subdoubleton #-}

    -- | A variant of 'link_' which makes sure to execute side-effects
    -- in the right order.
    linkA
        :: Applicative f
        => Prefix -> f (IntMap a)
        -> Prefix -> f (IntMap a)
        -> f (IntMap a)
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
        -> f (IntMap a)
        -> f (IntMap a)
        -> f (IntMap a)
    binA p m a b
      | m < 0     = liftA2 (flip (Bin p m)) b a
      | otherwise = liftA2       (Bin p m)  a b
    {-# INLINE binA #-}
{-# INLINE mergeA #-}


{--------------------------------------------------------------------
  Min\/Max
--------------------------------------------------------------------}

-- | \(O(\min(n,W))\). Update the value at the minimal key.
--
-- > updateMinWithKey (\ k a -> ((show k) ++ ":" ++ a)) (fromList [(5,"a"), (3,"b")]) == fromList [(3,"3:b"), (5,"a")]

updateMinWithKey :: (Key -> a -> a) -> IntMap a -> IntMap a
updateMinWithKey f t =
  case t of Bin p m l r | m < 0 -> Bin p m l (go f r)
            _ -> go f t
  where
    go f' (Bin p m l r) = Bin p m (go f' l) r
    go f' (Tip k y) = Tip k (f' k y)

-- | \(O(\min(n,W))\). Update the value at the maximal key.
--
-- > updateMaxWithKey (\ k a -> ((show k) ++ ":" ++ a)) (fromList [(5,"a"), (3,"b")]) == fromList [(3,"b"), (5,"5:a")]

updateMaxWithKey :: (Key -> a -> a) -> IntMap a -> IntMap a
updateMaxWithKey f t =
  case t of Bin p m l r | m < 0 -> Bin p m (go f l) r
            _ -> go f t
  where
    go f' (Bin p m l r) = Bin p m l (go f' r)
    go f' (Tip k y) = Tip k (f' k y)

-- | \(O(\min(n,W))\). The minimal key of the map.
findMin :: IntMap a -> (Key, a)
findMin (Tip k v) = (k,v)
findMin (Bin _ m l r)
  | m < 0     = go r
  | otherwise = go l
    where go (Tip k v)      = (k,v)
          go (Bin _ _ l' _) = go l'

-- | \(O(\min(n,W))\). The maximal key of the map.
findMax :: IntMap a -> (Key, a)
findMax (Tip k v) = (k,v)
findMax (Bin _ m l r)
  | m < 0     = go l
  | otherwise = go r
    where go (Tip k v)      = (k,v)
          go (Bin _ _ _ r') = go r'

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
isProperSubmapOfBy predicate t1 t2
  = case submapCmp predicate t1 t2 of
      LT -> True
      _  -> False

submapCmp :: (a -> b -> Bool) -> IntMap a -> IntMap b -> Ordering
submapCmp predicate t1@(Bin p1 m1 l1 r1) (Bin p2 m2 l2 r2)
  | shorter m1 m2  = GT
  | shorter m2 m1  = submapCmpLt
  | p1 == p2       = submapCmpEq
  | otherwise      = GT  -- disjoint
  where
    submapCmpLt | nomatch p1 p2 m2  = GT
                | zero p1 m2        = submapCmp predicate t1 l2
                | otherwise         = submapCmp predicate t1 r2
    submapCmpEq = case (submapCmp predicate l1 l2, submapCmp predicate r1 r2) of
                    (GT,_ ) -> GT
                    (_ ,GT) -> GT
                    (EQ,EQ) -> EQ
                    _       -> LT

submapCmp _         (Bin _ _ _ _) _  = GT
submapCmp predicate (Tip kx x) (Tip ky y)
  | (kx == ky) && predicate x y = EQ
  | otherwise                   = GT  -- disjoint
submapCmp predicate (Tip k x) t
  = case lookup k t of
     Just y | predicate x y -> LT
     _                      -> GT -- disjoint

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
isSubmapOfBy predicate t1@(Bin p1 m1 l1 r1) (Bin p2 m2 l2 r2)
  | shorter m1 m2  = False
  | shorter m2 m1  = match p1 p2 m2 &&
                       if zero p1 m2
                       then isSubmapOfBy predicate t1 l2
                       else isSubmapOfBy predicate t1 r2
  | otherwise      = (p1==p2) && isSubmapOfBy predicate l1 l2 && isSubmapOfBy predicate r1 r2
isSubmapOfBy _         (Bin _ _ _ _) _ = False
isSubmapOfBy predicate (Tip k x) t     = case lookup k t of
                                         Just y  -> predicate x y
                                         Nothing -> False

{--------------------------------------------------------------------
  Mapping
--------------------------------------------------------------------}
-- | \(O(n)\). Map a function over all values in the map.
--
-- > map (++ "x") (fromList [(5,"a"), (3,"b")]) == fromList [(3, "bx"), (5, "ax")]

map :: (a -> b) -> IntMap a -> IntMap b
map f = go
  where
    go (Bin p m l r) = Bin p m (go l) (go r)
    go (Tip k x)     = Tip k (f x)

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
mapWithKey f t
  = case t of
      Bin p m l r -> Bin p m (mapWithKey f l) (mapWithKey f r)
      Tip k x     -> Tip k (f k x)

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
traverseWithKey f = go
  where
    go (Tip k v) = Tip k <$> f k v
    go (Bin p m l r)
      | m < 0     = liftA2 (flip (Bin p m)) (go r) (go l)
      | otherwise = liftA2 (Bin p m) (go l) (go r)
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
      Bin p m l r
        | m < 0 ->
            let (a1,r') = mapAccumL f a r
                (a2,l') = mapAccumL f a1 l
            in (a2,Bin p m l' r')
        | otherwise  ->
            let (a1,l') = mapAccumL f a l
                (a2,r') = mapAccumL f a1 r
            in (a2,Bin p m l' r')
      Tip k x     -> let (a',x') = f a k x in (a',Tip k x')

-- | \(O(n)\). The function @'mapAccumRWithKey'@ threads an accumulating
-- argument through the map in descending order of keys.
mapAccumRWithKey :: (a -> Key -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccumRWithKey f a t
  = case t of
      Bin p m l r
        | m < 0 ->
            let (a1,l') = mapAccumRWithKey f a l
                (a2,r') = mapAccumRWithKey f a1 r
            in (a2,Bin p m l' r')
        | otherwise  ->
            let (a1,r') = mapAccumRWithKey f a r
                (a2,l') = mapAccumRWithKey f a1 l
            in (a2,Bin p m l' r')
      Tip k x     -> let (a',x') = f a k x in (a',Tip k x')

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
    Bin _ m l r
      | m < 0 -> go (go z l) r -- put negative numbers before
      | otherwise -> go (go z r) l
    _ -> go z t
  where
    go z' (Tip _ x)     = f x z'
    go z' (Bin _ _ l r) = go (go z' r) l
{-# INLINE foldr #-}

-- | \(O(n)\). A strict version of 'foldr'. Each application of the operator is
-- evaluated before using the result in the next application. This
-- function is strict in the starting value.
foldr' :: (a -> b -> b) -> b -> IntMap a -> b
foldr' f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    Bin _ m l r
      | m < 0 -> go (go z l) r -- put negative numbers before
      | otherwise -> go (go z r) l
    _ -> go z t
  where
    go !z' (Tip _ x)     = f x z'
    go  z' (Bin _ _ l r) = go (go z' r) l
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
    Bin _ m l r
      | m < 0 -> go (go z r) l -- put negative numbers before
      | otherwise -> go (go z l) r
    _ -> go z t
  where
    go !z' (Tip _ x)     = f z' x
    go  z' (Bin _ _ l r) = go (go z' l) r
{-# INLINE foldl #-}

-- | \(O(n)\). A strict version of 'foldl'. Each application of the operator is
-- evaluated before using the result in the next application. This
-- function is strict in the starting value.
foldl' :: (a -> b -> a) -> a -> IntMap b -> a
foldl' f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    Bin _ m l r
      | m < 0 -> go (go z r) l -- put negative numbers before
      | otherwise -> go (go z l) r
    _ -> go z t
  where
    go !z' (Tip _ x)     = f z' x
    go  z' (Bin _ _ l r) = go (go z' l) r
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
    Bin _ m l r
      | m < 0 -> go (go z l) r -- put negative numbers before
      | otherwise -> go (go z r) l
    _ -> go z t
  where
    go z' (Tip kx x)    = f kx x z'
    go z' (Bin _ _ l r) = go (go z' r) l
{-# INLINE foldrWithKey #-}

-- | \(O(n)\). A strict version of 'foldrWithKey'. Each application of the operator is
-- evaluated before using the result in the next application. This
-- function is strict in the starting value.
foldrWithKey' :: (Key -> a -> b -> b) -> b -> IntMap a -> b
foldrWithKey' f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    Bin _ m l r
      | m < 0 -> go (go z l) r -- put negative numbers before
      | otherwise -> go (go z r) l
    _ -> go z t
  where
    go !z' (Tip kx x)    = f kx x z'
    go  z' (Bin _ _ l r) = go (go z' r) l
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
    Bin _ m l r
      | m < 0 -> go (go z r) l -- put negative numbers before
      | otherwise -> go (go z l) r
    _ -> go z t
  where
    go z' (Tip kx x)    = f z' kx x
    go z' (Bin _ _ l r) = go (go z' l) r
{-# INLINE foldlWithKey #-}

-- | \(O(n)\). A strict version of 'foldlWithKey'. Each application of the operator is
-- evaluated before using the result in the next application. This
-- function is strict in the starting value.
foldlWithKey' :: (a -> Key -> b -> a) -> a -> IntMap b -> a
foldlWithKey' f z = \t ->      -- Use lambda t to be inlinable with two arguments only.
  case t of
    Bin _ m l r
      | m < 0 -> go (go z r) l -- put negative numbers before
      | otherwise -> go (go z l) r
    _ -> go z t
  where
    go !z' (Tip kx x)    = f z' kx x
    go  z' (Bin _ _ l r) = go (go z' l) r
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
    go (Tip kx x)    = f kx x
    go (Bin _ m l r)
      | m < 0     = go r `mappend` go l
      | otherwise = go l `mappend` go r
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
keysSet (Tip kx _) = IntSet.singleton kx
keysSet (Bin p m l r)
  | m .&. IntSet.suffixBitMask == 0 = IntSet.Bin p m (keysSet l) (keysSet r)
  | otherwise = IntSet.Tip (p .&. IntSet.prefixBitMask) (computeBm (computeBm 0 l) r)
  where computeBm !acc (Bin _ _ l' r') = computeBm (computeBm acc l') r'
        computeBm acc (Tip kx _) = acc .|. IntSet.bitmapOf kx

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
fromList [] = error "fromList: empty list"
fromList ((x0,a) : xs)
  = Foldable.foldl' ins (singleton x0 a) xs
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
  = Foldable.foldl' ins (errorWithoutStackTrace "fromListWithKey") xs
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
fromMonoListWithKey distinct f = go
  where
    go []              = error "fromMonoListWithKey: empty list"
    go ((kx,vx) : zs1) = addAll' kx vx zs1

    -- `addAll'` collects all keys equal to `kx` into a single value,
    -- and then proceeds with `addAll`.
    addAll' !kx vx []
        = Tip kx vx
    addAll' !kx vx ((ky,vy) : zs)
        | Nondistinct <- distinct, kx == ky
        = let v = f kx vy vx in addAll' ky v zs
        -- inlined: | otherwise = addAll kx (Tip kx vx) (ky : zs)
        | m <- branchMask kx ky
        , Inserted ty zs' <- addMany' m ky vy zs
        = addAll kx (linkWithMask m ky ty {-kx-} (Tip kx vx)) zs'

    -- for `addAll` and `addMany`, kx is /a/ key inside the tree `tx`
    -- `addAll` consumes the rest of the list, adding to the tree `tx`
    addAll !_kx !tx []
        = tx
    addAll !kx !tx ((ky,vy) : zs)
        | m <- branchMask kx ky
        , Inserted ty zs' <- addMany' m ky vy zs
        = addAll kx (linkWithMask m ky ty {-kx-} tx) zs'

    -- `addMany'` is similar to `addAll'`, but proceeds with `addMany'`.
    addMany' !_m !kx vx []
        = Inserted (Tip kx vx) []
    addMany' !m !kx vx zs0@((ky,vy) : zs)
        | Nondistinct <- distinct, kx == ky
        = let v = f kx vy vx in addMany' m ky v zs
        -- inlined: | otherwise = addMany m kx (Tip kx vx) (ky : zs)
        | mask kx m /= mask ky m
        = Inserted (Tip kx vx) zs0
        | mxy <- branchMask kx ky
        , Inserted ty zs' <- addMany' mxy ky vy zs
        = addMany m kx (linkWithMask mxy ky ty {-kx-} (Tip kx vx)) zs'

    -- `addAll` adds to `tx` all keys whose prefix w.r.t. `m` agrees with `kx`.
    addMany !_m !_kx tx []
        = Inserted tx []
    addMany !m !kx tx zs0@((ky,vy) : zs)
        | mask kx m /= mask ky m
        = Inserted tx zs0
        | mxy <- branchMask kx ky
        , Inserted ty zs' <- addMany' mxy ky vy zs
        = addMany m kx (linkWithMask mxy ky ty {-kx-} tx) zs'
{-# INLINE fromMonoListWithKey #-}

data Inserted a = Inserted !(IntMap a) ![(Key,a)]

data Distinct = Distinct | Nondistinct

{--------------------------------------------------------------------
  Eq
--------------------------------------------------------------------}
instance Eq a => Eq (IntMap a) where
  t1 == t2  = equal t1 t2
  t1 /= t2  = nequal t1 t2

equal :: Eq a => IntMap a -> IntMap a -> Bool
equal (Bin p1 m1 l1 r1) (Bin p2 m2 l2 r2)
  = (m1 == m2) && (p1 == p2) && (equal l1 l2) && (equal r1 r2)
equal (Tip kx x) (Tip ky y)
  = (kx == ky) && (x==y)
equal _   _   = False

nequal :: Eq a => IntMap a -> IntMap a -> Bool
nequal (Bin p1 m1 l1 r1) (Bin p2 m2 l2 r2)
  = (m1 /= m2) || (p1 /= p2) || (nequal l1 l2) || (nequal r1 r2)
nequal (Tip kx x) (Tip ky y)
  = (kx /= ky) || (x/=y)
nequal _   _   = True

-- | @since 0.5.9
instance Eq1 IntMap where
  liftEq eq (Bin p1 m1 l1 r1) (Bin p2 m2 l2 r2)
    = (m1 == m2) && (p1 == p2) && (liftEq eq l1 l2) && (liftEq eq r1 r2)
  liftEq eq (Tip kx x) (Tip ky y)
    = (kx == ky) && (eq x y)
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
    a <$ Bin p m l r = Bin p m (a <$ l) (a <$ r)
    a <$ Tip k _     = Tip k a
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
  Link
--------------------------------------------------------------------}
link :: Prefix -> IntMap a -> Prefix -> IntMap a -> IntMap a
link p1 t1 p2 t2 = linkWithMask (branchMask p1 p2) p1 t1 {-p2-} t2
{-# INLINE link #-}

-- `linkWithMask` is useful when the `branchMask` has already been computed
linkWithMask :: Mask -> Prefix -> IntMap a -> IntMap a -> IntMap a
linkWithMask m p1 t1 {-p2-} t2
  | zero p1 m = Bin p m t1 t2
  | otherwise = Bin p m t2 t1
  where
    p = mask p1 m
{-# INLINE linkWithMask #-}

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
    x@(Tip _ _) -> [x]
    Bin _ m l r | m < 0 -> [r, l]
                | otherwise -> [l, r]
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
  Bin p m l r ->
    showsTree wide (withBar rbars) (withEmpty rbars) r .
    showWide wide rbars .
    showsBars lbars . showString (showBin p m) . showString "\n" .
    showWide wide lbars .
    showsTree wide (withEmpty lbars) (withBar lbars) l
  Tip k x ->
    showsBars lbars .
    showString " " . shows k . showString ":=" . shows x . showString "\n"

showsTreeHang :: Show a => Bool -> [String] -> IntMap a -> ShowS
showsTreeHang wide bars t = case t of
  Bin p m l r ->
    showsBars bars . showString (showBin p m) . showString "\n" .
    showWide wide bars .
    showsTreeHang wide (withBar bars) l .
    showWide wide bars .
    showsTreeHang wide (withEmpty bars) r
  Tip k x ->
    showsBars bars .
    showString " " . shows k . showString ":=" . shows x . showString "\n"

showBin :: Prefix -> Mask -> String
showBin _ _
  = "*" -- ++ show (p,m)

showWide :: Bool -> [String] -> String -> String
showWide wide bars
  | wide      = showString (concat (reverse bars)) . showString "|\n"
  | otherwise = id

showsBars :: [String] -> ShowS
showsBars bars
  = case bars of
      [] -> id
      _ : tl -> showString (concat (reverse tl)) . showString node

node :: String
node = "+--"

withBar, withEmpty :: [String] -> [String]
withBar bars   = "|  ":bars
withEmpty bars = "   ":bars
